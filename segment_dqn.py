import copy
import math
import torch
import tqdm
from losses import RDQNSegmentLoss
import popgym
from utils import (
    apply_primers,
    get_modules,
    hard_update,
    soft_update,
    load_popgym_env,
    load_wrapped_popgym_env,
    truncate_trajectories,
)
import numpy as np
import gymnasium as gym
import importlib
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    EnsembleModule,
)
from torch import nn
from torchrl.collectors import SyncDataCollector
import tensordict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import ConvNet, EGreedyWrapper, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, HardUpdate
from updater import SoftUpdateModule
import wandb
import yaml
import argparse



a = argparse.ArgumentParser()
a.add_argument("config", type=str)
a.add_argument("--seed", "-s", type=int, default=0)
a.add_argument("--device", "-d", type=str, default="cpu")
args = a.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config["seed"] = args.seed
config["device"] = args.device
# config["epochs"] = config["total_frames"] // config["segment_length"]
config["total_frames"] = config["epochs"] * config["segment_length"]
config["total_random_frames"] = config["random_epochs"] * config["segment_length"]

torch.manual_seed(config["seed"])

device = torch.device(config["device"])
env = load_wrapped_popgym_env(config, device)
eval_env = load_wrapped_popgym_env(config, device, eval=True)


# state_mod, q_mod, target_state_mod, target_q_mod= get_modules(config, env, training=True)
# policy_state_mod, policy_q_mod, _, _ = get_modules(config, env, training=False)
module = get_modules(config, env, training=True)
eval_policy = get_modules(config, env, training=False)
env = apply_primers(env, module[1])
eval_env = apply_primers(eval_env, module[1])
#eval_policy = Seq(policy_state_mod, policy_q_mod)
policy = EGreedyWrapper(
    eval_policy,
    annealing_num_steps=config["epochs"],
    spec=env.action_spec,
    eps_init=config["eps_init"],
    eps_end=config["eps_end"],
)

loss_fn = RDQNSegmentLoss(
    module,
    delay_value=True,
    action_space=env.action_spec,
)
# loss_fn = RDQNSegmentLoss(
#     state_mod,
#     target_state_mod,
#     q_mod,
#     target_q_mod,
#     env.action_spec,
    
# )
updater = HardUpdate(loss_fn, value_network_update_interval=config["target_delay"])



optim = torch.optim.AdamW(
    module.parameters(), lr=config["lr"]
    #[*pre.parameters(), *lstm.parameters(), *post.parameters()], lr=config["lr"]
    #[*state_mod.parameters(), *q_mod.parameters()], lr=config["lr"]
)
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=config["segment_length"],
    total_frames=config["total_frames"],
    split_trajs=True,
    init_random_frames=config["total_random_frames"],
)
eval_collector = SyncDataCollector(
    eval_env,
    eval_policy,
    frames_per_batch=config["eval_length"],
    total_frames=config["total_frames"],
    # split_trajs=True,
    init_random_frames=0,
    reset_at_each_iter=True,
)
rb = TensorDictReplayBuffer(
    storage=LazyTensorStorage(config["buffer_size"]),
    batch_size=config["batch_size"],
    prefetch=config["batch_size"] * config["utd"],
)
pbar = tqdm.tqdm(total=config["epochs"] + config["random_epochs"])
longest = 0

best_reward = -float("inf")
eval_reward = -float("inf")
best_eval_reward = -float("inf")
nonzero_frames = 0

if config["wandb"]:
    wandb.init(project="rdqn_segment", config=config)
for i, data in enumerate(collector, 1):
    frames_this_iter = data[("collector", "mask")].sum()
    nonzero_frames += frames_this_iter
    to_log = {
        "epoch": i,
        "collector/nonzero_frames": nonzero_frames,
    }
    pbar.update()
    # it is important to pass data that is not flattened
    # TODO: data is often longer than this, what does padding do?
    # truncate_trajectories(data, config["segment_length"])
    # Do not store recurrent states as we want zero-states
    data = data.exclude(*module.recurrent_keys)
    data['next'] = data['next'].exclude(*module.recurrent_keys)
    padded = tensordict.pad(data, [0, 0, 0, config["segment_length"] - data.shape[-1]])
    rb.extend(padded.cpu())
    unpadded_data = data.masked_select(data[("collector", "mask")])
    if collector._frames < config["total_random_frames"]:
        continue
    mean_reward = unpadded_data["episode_reward"][
        unpadded_data[("next", "done")]
    ].mean()
    if mean_reward > best_reward:
        best_reward = mean_reward
    for u in range(config["utd"]):
        if i % config["reset_interval"] == 0:
            module.reset_parameters_recursive()

        batch = rb.sample().to(device)
        loss = loss_fn(batch)
        # # Do not use stale states
        # # del batch['recurrent_state_c'], batch['recurrent_state_h'], batch[('next', 'recurrent_state_c')], batch[('next', 'recurrent_state_h')]
        # state_mod(batch)
        # # Prevent leaking learned network states into the target network
        # # Otherwise s_0 from learned net will feed into target network at initial timestep
        # # del (
        # #     batch[("next", "recurrent_state_s")],
        # #     batch[("next", "recurrent_state_z")],
        # # )
        # with torch.no_grad():
        #     target_state_mod(batch["next"])

        # # Replace stored states as target will have overwritten them
        # # batch.update(stored_states)
        # # Compute markov_state
        # mask = batch[("collector", "mask")]
        # masked = batch.masked_select(mask)
        # loss = loss_fn(masked)

        loss["loss"].backward()
        optim.step()
        optim.zero_grad()

        # soft_update(target_state_mod, state_mod, config["tau"])
        # hard_update(
        #     target_state_mod, state_mod, config["target_delay"], i * config["utd"] + u
        # )
        # hard_update(
        #     target_q_mod, q_mod, config["target_delay"], i * config["utd"] + u
        # )
        updater.step()

    pbar.set_description(
        f"loss: {loss['loss'].item():.3f}, rew(c:{mean_reward:.2f}, b:{best_reward:.2f}), eval rew: (c:{eval_reward:.2f}, b:{best_eval_reward:.2f})"
    )
    if not math.isnan(mean_reward):
        to_log = {**to_log, "train/reward": mean_reward}

    to_log = {
        **to_log,
        "train/loss": loss["loss"].item(),
        "train/best_reward": best_reward,
        "train/epsilon": policy.eps,
        "train/buffer_size": len(rb),
        "train/buffer_capacity": len(rb) / config["buffer_size"],
        "collector/action_histogram": wandb.Histogram(data["action"].cpu()),
    }
    policy.step(1)
    collector.policy.module.load_state_dict(module.module.state_dict())

    if i % config["eval_interval"] == 0:
        eval_collector.policy.load_state_dict(module.state_dict())
        rollout = next(eval_collector.iterator())

        # unpadded_rollout = data.masked_select(data[('collector', 'mask')])
        # eval_reward = unpadded_rollout["episode_reward"][unpadded_rollout[("next", "done")]].mean()
        eval_reward = rollout["episode_reward"][rollout[("next", "done")]].mean()
        if eval_reward > best_eval_reward:
            best_eval_reward = eval_reward
        to_log = {
            **to_log,
            "eval/reward": eval_reward,
            "eval/best_reward": best_eval_reward,
            "eval/num_episodes": rollout[('next', 'done')].sum(),
        }

    if config["wandb"]:
        wandb.log(to_log)
