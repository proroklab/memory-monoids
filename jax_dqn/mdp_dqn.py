from functools import partial
from jax import numpy as jnp
from jax import vmap, jit, random
import jax
import numpy as np
from jax import random, vmap, nn

# from cpprb import ReplayBuffer
from buffer import ReplayBuffer

# import flax
# from flax import linen as nn
import equinox as eqx
from losses import stream_dqn_loss, segment_dqn_loss
from modules import greedy_policy
from modules import hard_update, soft_update, mean_noise
#from collector import SegmentCollector, StreamCollector
from tape_collector import TapeCollector
import optax
import tqdm
import argparse
import yaml

from modules import epsilon_greedy_policy, anneal
from linear_transformer import LTQNetwork
from gru import GRUQNetwork
from mlp import MLPQNetwork
from utils import load_popgym_env
from losses import segment_dqn_loss, segment_constrained_dqn_loss

model_map = {MLPQNetwork.name: MLPQNetwork, GRUQNetwork.name: GRUQNetwork, LTQNetwork.name: LTQNetwork}

a = argparse.ArgumentParser()
a.add_argument("config", type=str)
a.add_argument("--seed", "-s", type=int, default=None)
a.add_argument("--device", "-d", type=str, default="cpu")
a.add_argument("--wandb", "-w", action="store_true")
args = a.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if args.seed is not None:
    config["seed"] = args.seed
config["eval"]["seed"] = config["seed"] + 1000
config["device"] = args.device

if args.wandb:
    import wandb

    wandb.init(project="jax_dqn", config=config)

env = load_popgym_env(config, popgym=False)
eval_env = load_popgym_env(config, eval=True, popgym=False)
obs_shape = env.observation_space.shape[0]
act_shape = env.action_space.n

key = random.PRNGKey(config["seed"])
eval_key = random.PRNGKey(config["eval"]["seed"])
eval_keys = random.split(eval_key, config["eval"]["episodes"])
lr_schedule = optax.cosine_decay_schedule(
    init_value=config["train"]["lr"], 
    decay_steps=config['collect']['epochs'],
)
opt = optax.adamw(lr_schedule, weight_decay=0.001)


rb = ReplayBuffer(
    config["train"]["buffer_size"],
    {
        "observation": {
            "shape": obs_shape,
            "dtype": jnp.float32,
        },
        "action": {"shape": (), "dtype": jnp.int32},
        "reward": {"shape": (), "dtype": jnp.float32},
        "next_observation": {
            "shape": obs_shape,
            "dtype": jnp.float32,
        },
        "start": {"shape": (), "dtype": bool},
        "done": {"shape": (), "dtype": bool},
    },
)


key, *keys = random.split(key, 3)
model_class = model_map[config["model"]["name"]]
q_network = model_class(obs_shape, act_shape, config["model"], keys[0])
q_target = model_class(obs_shape, act_shape, config["model"], keys[0])
opt_state = opt.init(eqx.filter(q_network, eqx.is_inexact_array))
epochs = config["collect"]["random_epochs"] + config["collect"]["epochs"]
pbar = tqdm.tqdm(total=epochs)
best_eval_ep_reward = best_cumulative_reward = eval_ep_reward = -np.inf
need_reset = True
collector = TapeCollector(env, config)
eval_collector = TapeCollector(eval_env, config["eval"])
transitions_collected = 0
transitions_trained = 0

epoch_keys = random.split(key, epochs + 1)
key, epoch_keys = epoch_keys[0], epoch_keys[1:]
collect_keys = random.split(key, epochs + 1)
key, collect_keys = collect_keys[0], collect_keys[1:]
loss_keys = random.split(key, epochs + 1)
key, loss_keys = loss_keys[0], loss_keys[1:]

for epoch in range(1, epochs + 1):
    pbar.update()
    progress = max(
        0, (epoch - config["collect"]["random_epochs"]) / config["collect"]["epochs"]
    )
    (
        transitions,
        cumulative_reward,
        best_cumulative_reward
    ) = collector(q_network, epsilon_greedy_policy, progress, epoch_keys[epoch-1], False)

    # Remove zeros as segment collector will pad
    rb.add(
        **transitions
    )
    rb.on_episode_end()
    transitions_collected += len(transitions['done'])

    if epoch <= config["collect"]["random_epochs"]:
        continue

    data = rb.sample(config["train"]["batch_size"], collect_keys[epoch-1])
    transitions_trained += len(data["done"])
    outputs, gradient = stream_dqn_loss(
        q_network, q_target, data, config["train"]["gamma"], key=loss_keys[epoch-1]
    )
    loss, (q_mean, target_mean, target_network_mean) = outputs
    updates, opt_state = opt.update(
        gradient, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=1 / config["train"]["target_delay"])

    # Eval
    if epoch % config["eval"]["interval"] == 0:
        _, eval_ep_reward, best_eval_ep_reward = eval_collector(
            q_network, greedy_policy, 1.0, eval_key, True
        )

    to_log = {
        "collect/epoch": epoch,
        "collect/train_epoch": max(0, epoch - config["collect"]["random_epochs"]),
        "collect/reward": cumulative_reward,
        "collect/best_reward": best_cumulative_reward,
        "collect/buffer_capacity": rb.get_stored_size()
        / config["train"]["buffer_size"],
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
        "train/grad_unclipped_global_norm": optax.global_norm(gradient),
    }
    to_log = {k: v for k, v in to_log.items() if jnp.isfinite(v)}
    if args.wandb:
        wandb.log(to_log)

    pbar.set_description(
        f"eval: {eval_ep_reward:.2f}, {best_eval_ep_reward:.2f} "
        + f"train: {cumulative_reward:.2f}, {best_cumulative_reward:.2f} "
        + f"loss: {loss:.3f} "
        + f"eps: {anneal(config['collect']['eps_start'], config['collect']['eps_end'], progress):.2f} "
        + f"buf: {rb.get_stored_size() / config['train']['buffer_size']:.2f} "
        + f"qm: {q_mean:.2f} "
        + f"tm: {target_mean:.2f} "
    )
