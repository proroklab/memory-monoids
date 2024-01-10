from functools import partial
from jax import numpy as jnp
from jax import vmap, jit, random
import jax
import numpy as np
from jax import random, vmap, nn
import time

# from cpprb import ReplayBuffer
from buffer import TapeBuffer, ShuffledTapeBuffer
from collector.tape_collector import TapeCollector

# import flax
# from flax import linen as nn
import equinox as eqx
from modules import greedy_policy, hard_update, soft_update, RecurrentQNetwork
import optax
import tqdm
import argparse
import yaml

from modules import epsilon_greedy_policy, anneal, boltzmann_policy, mean_noise
from memory.gru import GRU
from memory.sffm import SFFM, NSFFM
from memory.ffm import FFM
from memory.linear_transformer import LinearAttention, StackedLinearAttention
from memory.lru import StackedLRU
from memory.s5 import StackedS5

from utils import get_wandb_model_info, load_popgym_env
from losses import tape_dqn_loss, tape_dqn_loss_filtered, tape_ddqn_loss, tape_update

model_map = {GRU.name: GRU, SFFM.name: SFFM, NSFFM.name: NSFFM, FFM.name: FFM, LinearAttention.name: LinearAttention, StackedLinearAttention.name: StackedLinearAttention, StackedLRU.name: StackedLRU, StackedS5.name: StackedS5}

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
    config["train"]["ratio"] = 1
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
# lr_schedule = optax.linear_schedule(
#     init_value=0,
#     end_value=config["train"]["lr"], 
#     transition_steps=config["train"]["warmup_epochs"] * config["train"]["train_ratio"],
# )

opt = optax.chain(
    optax.zero_nans(),
    optax.clip_by_global_norm(config["train"]["gradient_scale"]),
    optax.adamw(lr_schedule, weight_decay=config["train"]["weight_decay"], eps=config["train"]["adam_eps"])
)


rb = ShuffledTapeBuffer(
    config["buffer"]["size"],
    "start",
    {
        "observation": {
            "shape": obs_shape,
            "dtype": np.float32,
        },
        "action": {"shape": (), "dtype": np.int32},
        "next_reward": {"shape": (), "dtype": np.float32},
        "next_observation": {
            "shape": obs_shape,
            "dtype": np.float32,
        },
        "start": {"shape": (), "dtype": bool},
        "next_terminated": {"shape": (), "dtype": bool},
        "next_truncated": {"shape": (), "dtype": bool},
        "next_done": {"shape": (), "dtype": bool},
        "episode_id": {"shape": (), "dtype": np.int64},
    },
    swap_iters=config["buffer"]["swap_iters"],
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
collector = TapeCollector(env, config)
eval_collector = TapeCollector(eval_env, config["eval"])
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
        q_network, q_target, opt_state, q_mean, target_mean, target_network_mean, error_min, error_max, loss, gradient = eqx.filter_jit(tape_update)(q_network, q_target, data, opt, opt_state, gamma, 1 / config["train"]["target_delay"], loss_key)

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
            jac = eqx.filter_jit(eqx.filter_grad(tape_dqn_loss_filtered))(eval_transitions["observation"], q_eval, q_target, eval_transitions, gamma, eval_key)
            temporal_grad = jac.sum(-1)
            grad_info = {"grads/terminal_dloss_dx": temporal_grad}


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
    #to_log = {k: v for k, v in to_log.items() if jnp.isfinite(v)}

    pbar.set_description(
        f"eval: {eval_ep_reward:.2f}, {best_eval_ep_reward:.2f} "
        + f"train: {cumulative_reward:.2f}, {best_ep_reward:.2f} "
        + f"loss: {loss:.3f} "
        + f"eps: {anneal(config['collect']['eps_start'], config['collect']['eps_end'], progress):.2f} "
        + f"buf: {rb.get_stored_size() / config['buffer']['size']:.2f} "
        + f"qm: {q_mean:.2f} "
        + f"tm: {target_mean:.2f} "
    )
