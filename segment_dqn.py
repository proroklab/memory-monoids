from functools import partial
import time
from jax import numpy as jnp
from jax import vmap, jit, random
import jax
import numpy as np
from jax import random, vmap, nn

from buffer import ReplayBuffer

import equinox as eqx
from collector.segment_collector import SegmentCollector
import optax
import tqdm
import argparse
import yaml
from memory.ffm import FFM
from memory.linear_transformer import LinearAttention

from modules import epsilon_greedy_policy, anneal, RecurrentQNetwork, hard_update, soft_update, greedy_policy
from memory.gru import GRU
from memory.sffm import NSFFM, SFFM
from utils import get_wandb_model_info, load_popgym_env, scale_by_norm
from losses import segment_ddqn_loss, segment_dqn_loss

model_map = {GRU.name: GRU, SFFM.name: SFFM, NSFFM.name: NSFFM, FFM.name: FFM, LinearAttention.name: LinearAttention}

a = argparse.ArgumentParser()
a.add_argument("config", type=str)
a.add_argument("--seed", "-s", type=int, default=None)
a.add_argument("--debug", "-d", action="store_true")
a.add_argument("--wandb", "-w", action="store_true")
a.add_argument('--name', '-n', type=str, default=None)
a.add_argument('--project', '-p', type=str, default="jax_dqn")
a.add_argument('--load', '-l', type=str, default=None)
a.add_argument('--log-model', '-m', action="store_true")
a.add_argument('--log-grads', '-g', action="store_true")
args = a.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if args.debug:
    config["collect"]["random_epochs"] = 10
    config["train"]["batch_size"] = 10
    jax.config.update('jax_disable_jit', True)

if args.seed is not None:
    config["seed"] = args.seed
config["eval"]["seed"] = config["seed"] + 1000

if args.wandb:
    import wandb

    wandb.init(project=args.project, name=args.name, config=config)

env = load_popgym_env(config)
eval_env = load_popgym_env(config, eval=True)
obs_shape = env.observation_space.shape[0]
act_shape = env.action_space.n

key = random.PRNGKey(config["seed"])
eval_key = random.PRNGKey(config["eval"]["seed"])
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0,
    peak_value=config["train"]["lr_start"], 
    warmup_steps=config["train"]["warmup_epochs"],
    decay_steps=config['collect']['epochs'] * config["train"]["train_ratio"],
    end_value=config["train"]["lr_end"]
)

opt = optax.chain(
    optax.zero_nans(),
    optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(lr_schedule, weight_decay=config["train"]["weight_decay"], eps=config["train"]["adam_eps"])
)


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
        "next_done": {"shape": config["collect"]["segment_length"], "dtype": bool},
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
q_target = eqx.tree_inference(RecurrentQNetwork(obs_shape, act_shape, memory_target, config["model"], model_key), True)
opt_state = opt.init(eqx.filter(q_network, eqx.is_inexact_array))
epochs = config["collect"]["random_epochs"] + config["collect"]["epochs"]
pbar = tqdm.tqdm(total=epochs)
best_eval_ep_reward = best_ep_reward = eval_ep_reward = ep_reward = -np.inf
collector = SegmentCollector(env, config)
eval_collector = SegmentCollector(eval_env, config["eval"])
model_info = eqx.filter_jit(get_wandb_model_info)(q_network) if args.log_model else {}
grad_info = {}
transitions_collected = 0
transitions_trained = 0

total_train_time = 0
train_elapsed = 0
total_train_time = 0
gamma = jnp.array(config["train"]["gamma"])
for epoch in range(1, epochs + 1):
    if epoch > 1:
        train_start = time.time()
    pbar.update()
    progress = jnp.array(max(
        0, (epoch - config["collect"]["random_epochs"]) / config["collect"]["epochs"]
    ))
    for _ in range(config["collect"]["ratio"]):
        key, epoch_key, sample_key, loss_key = random.split(key, 4)
        (
            transitions,
            cumulative_reward,
            best_ep_reward
        ) = collector(q_network, eqx.filter_jit(epsilon_greedy_policy), jnp.array(progress), epoch_key, False)

        rb.add(epoch_key, **transitions)
        rb.on_episode_end()
        if epoch <= config["collect"]["random_epochs"]:
            break
        transitions_collected += len(transitions['next_reward'])

    if epoch <= config["collect"]["random_epochs"]:
        continue


    for _ in range(config["train"]["train_ratio"]):
        _, sample_key = random.split(sample_key)
        data = rb.sample(config["train"]["batch_size"], sample_key)
        transitions_trained += len(data['next_reward'])

        
        outputs, gradient = eqx.filter_jit(eqx.filter_value_and_grad(segment_dqn_loss, has_aux=True))(
            q_network, q_target, data, gamma, loss_key
        )
        loss, (q_mean, target_mean, target_network_mean, error_min, error_max) = outputs
        updates, opt_state = eqx.filter_jit(opt.update)(
            gradient, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
        )
        q_network = eqx.filter_jit(eqx.apply_updates)(q_network, updates)
        q_target = eqx.filter_jit(soft_update)(q_network, q_target, tau=1 / config["train"]["target_delay"])

    if epoch > config["collect"]["random_epochs"] + 1:
        train_elapsed = time.time() - train_start
        total_train_time += train_elapsed
    # Eval
    if epoch % config["eval"]["interval"] == 0 or best_eval_ep_reward == -jnp.inf:
        q_eval = eqx.filter_jit(eqx.tree_inference)(q_network, True)
        model_info = eqx.filter_jit(get_wandb_model_info)(q_network) if args.log_model else {}
        eval_keys = random.split(eval_key, config["eval"]["episodes"])
        eval_rewards = []
        for i in range(config["eval"]["episodes"]):
            eval_transitions, eval_ep_reward, _ = eval_collector(
                q_eval, eqx.filter_jit(greedy_policy), 1.0, eval_keys[i], True
            )
            eval_rewards.append(eval_ep_reward)
        eval_ep_reward = np.mean(eval_rewards)
        if eval_ep_reward > best_eval_ep_reward:
            best_eval_ep_reward = eval_ep_reward

        # Compute BPTT grads
        if args.log_grads:
            raise NotImplementedError()


    if args.wandb:
#        action_hist = np.histogram(transitions["action"], bins=np.arange(act_shape + 1))
#        action_hist = wandb.Histogram(np_histogram=action_hist)

        to_log = {
            **{k: v.item() for k, v in model_info.items() if args.log_model},
            **grad_info,
            "collect/epoch": epoch,
            #"collect/action_hist": action_hist,
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
            "train/progress": progress,
            "train/q_mean": q_mean,
            "train/target_mean": target_mean,
            "train/target_network_mean": target_network_mean,
            "train/transitions": transitions_trained,
            "train/grad_global_norm": optax.global_norm(gradient),
            "train/time_this_epoch": train_elapsed,
            "train/time_total": total_train_time,
            "train/error_min": error_min,
            "train/error_max": error_max,
            "train/utd": transitions_trained / transitions_collected,
            "train/gamma": gamma,
        }
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
