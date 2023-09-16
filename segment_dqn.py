from functools import partial
from jax import numpy as jnp
from jax import vmap, jit, random
import jax
import numpy as np
from jax import random, vmap, nn

from buffer import ReplayBuffer

import equinox as eqx
from collector.segment_collector import BatchedSegmentCollector
import optax
import tqdm
import argparse
import yaml

from modules import epsilon_greedy_policy, anneal, RecurrentQNetwork, hard_update, soft_update, greedy_policy
from memory.gru import GRU
from memory.sffm import SFFM
from utils import load_popgym_env
from losses import segment_ddqn_loss

model_map = {GRU.name: GRU, SFFM.name: SFFM}

a = argparse.ArgumentParser()
a.add_argument("config", type=str)
a.add_argument("--seed", "-s", type=int, default=None)
a.add_argument("--debug", "-d", action="store_true")
a.add_argument("--wandb", "-w", action="store_true")
a.add_argument('--name', '-n', type=str, default=None)
args = a.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if args.debug:
    config["collect"]["random_epochs"] = 0
    jax.config.update('jax_disable_jit', True)

if args.seed is not None:
    config["seed"] = args.seed
config["eval"]["seed"] = config["seed"] + 1000

if args.wandb:
    import wandb

    wandb.init(project="jax_segment_dqn", name=args.name, config=config)

env = load_popgym_env(config)
eval_env = load_popgym_env(config, eval=True)
obs_shape = env.observation_space.shape[0]
act_shape = env.action_space.n

key = random.PRNGKey(config["seed"])
eval_key = random.PRNGKey(config["eval"]["seed"])
lr_schedule = optax.cosine_decay_schedule(
    init_value=config["train"]["lr"], 
    decay_steps=config['collect']['epochs'],
)
opt = optax.adamw(lr_schedule, weight_decay=0.001)


rb = ReplayBuffer(
    config["buffer"]["size"],
    {
        "observation": {
            "shape": (config["collect"]["segment_length"], obs_shape),
            "dtype": np.float32,
        },
        "action": {"shape": config["collect"]["segment_length"], "dtype": np.int32},
        "next_reward": {"shape": config["collect"]["segment_length"], "dtype": np.float32},
        "next_observation": {
            "shape": (config["collect"]["segment_length"], obs_shape),
            "dtype": np.float32,
        },
        "start": {"shape": config["collect"]["segment_length"], "dtype": bool},
        #"next_start": {"shape": config["collect"]["segment_length"], "dtype": bool},
        #"done": {"shape": config["collect"]["segment_length"], "dtype": bool},
        "next_terminated": {"shape": config["collect"]["segment_length"], "dtype": bool},
        "next_truncated": {"shape": config["collect"]["segment_length"], "dtype": bool},
        "mask": {"shape": config["collect"]["segment_length"], "dtype": bool},
        "episode_id": {"shape": config["collect"]["segment_length"], "dtype": np.int64},
    },
    config["buffer"]["contiguous"]
)


key, model_key, memory_key = random.split(key, 3)
memory_class = model_map[config["model"]["memory_name"]]
memory_network = memory_class(**config["model"]["memory"], key=memory_key)
memory_target = memory_class(**config["model"]["memory"], key=memory_key)
q_network = RecurrentQNetwork(obs_shape, act_shape, memory_network, config["model"], model_key)
q_target = RecurrentQNetwork(obs_shape, act_shape, memory_target, config["model"], model_key)
opt_state = opt.init(eqx.filter(q_network, eqx.is_inexact_array))
epochs = config["collect"]["random_epochs"] + config["collect"]["epochs"]
pbar = tqdm.tqdm(total=epochs)
best_eval_ep_reward = best_ep_reward = eval_ep_reward = ep_reward = -np.inf
collector = BatchedSegmentCollector(env, config)
eval_collector = BatchedSegmentCollector(eval_env, config["eval"])
transitions_collected = 0
transitions_trained = 0

for epoch in range(1, epochs + 1):
    pbar.update()
    progress = max(
        0, (epoch - config["collect"]["random_epochs"]) / config["collect"]["epochs"]
    )
    key, epoch_key, sample_key, loss_key = random.split(key, 4)
    (
        transitions,
        cumulative_reward,
        best_ep_reward
    ) = collector(q_network, epsilon_greedy_policy, jnp.array(progress), epoch_key, False)

    rb.add(**transitions)
    rb.on_episode_end()
    transitions_collected += transitions['mask'].sum()

    if epoch <= config["collect"]["random_epochs"]:
        continue
    
    data = rb.sample(config["train"]["batch_size"], sample_key)

    transitions_trained += data["mask"].sum()

    outputs, gradient = segment_ddqn_loss(
        q_network, q_target, data, config["train"]["gamma"], loss_key
    )
    loss, (q_mean, target_mean, target_network_mean) = outputs
    updates, opt_state = opt.update(
        gradient, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    # if epoch % config["train"]["target_delay"] == 0:
    #     q_target = hard_update(q_network, q_target)
    q_target = soft_update(q_network, q_target, tau=1 / config["train"]["target_delay"])

    # Eval
    if epoch % config["eval"]["interval"] == 0:
        eval_keys = random.split(eval_key, config["eval"]["episodes"])
        eval_rewards = []
        for i in range(config["eval"]["episodes"]):
            _, eval_ep_reward, _ = eval_collector(
                eqx.tree_inference(q_network, True), greedy_policy, 1.0, eval_keys[i], True
            )
            eval_rewards.append(eval_ep_reward)
        eval_ep_reward = np.mean(eval_rewards)
        if eval_ep_reward > best_eval_ep_reward:
            best_eval_ep_reward = eval_ep_reward

    # if epoch % 200:
    #     q_network.__call__.clear_caches()
    #     q_target.__call__.clear_caches()

    to_log = {
        "collect/epoch": epoch,
        "collect/train_epoch": max(0, epoch - config["collect"]["random_epochs"]),
        "collect/reward": cumulative_reward,
        "collect/best_reward": best_ep_reward,
        "collect/buffer_capacity": rb.get_stored_size()
        / config["buffer"]["size"],
        "collect/buffer_density": rb.get_density(),
        "collect/transitions": transitions_collected,
        "eval/collect/reward": eval_ep_reward,
        "eval/collect/best_reward": best_eval_ep_reward,
        "train/loss": loss,
        "train/epsilon": anneal(
            config["collect"]["eps_start"], config["collect"]["eps_end"], progress
        ),
        "train/q_mean": q_mean,
        "train/target_mean": target_mean,
        "train/target_network_mean": target_network_mean,
        "train/transitions": transitions_trained,
        "train/grad_global_norm": optax.global_norm(gradient),
        #"train/mean_noise": mean_noise(q_network),
    }
    to_log = {k: v for k, v in to_log.items() if jnp.isfinite(v)}
    if args.wandb:
        wandb.log(to_log)

    pbar.set_description(
        f"eval: {eval_ep_reward:.2f}, {best_eval_ep_reward:.2f} "
        + f"train: {cumulative_reward:.2f}, {best_ep_reward:.2f} "
        + f"loss: {loss:.3f} "
        + f"eps: {anneal(config['collect']['eps_start'], config['collect']['eps_end'], progress):.2f} "
        + f"buf: {rb.get_stored_size() / config['buffer']['size']:.2f} "
        + f"qm: {q_mean:.2f} "
        + f"tm: {target_mean:.2f} "
    )
