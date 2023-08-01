import copy
import math
import torch
import tqdm
import popgym
from utils import hard_update, soft_update, load_popgym_env, load_wrapped_popgym_env, truncate_trajectories
import numpy as np
import gymnasium as gym
import importlib
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn
from torchrl.collectors import SyncDataCollector
import tensordict
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import ConvNet, EGreedyWrapper, LSTMModule, MLP, QValueModule
from torchrl.objectives import DQNLoss, SoftUpdate, HardUpdate
from updater import SoftUpdateModule
import wandb
import yaml


with open("cartpole.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

torch.manual_seed(config["seed"])
config["epochs"] = config["total_frames"] // config["segment_length"]

device = torch.device(0) if torch.cuda.device_count() else torch.device("cpu")
env = load_wrapped_popgym_env(config, device)
eval_env = load_wrapped_popgym_env(config, device, eval=True)

# Networks
pre = nn.Sequential(
    nn.Linear(env.observation_spec['observation'].shape[0], config["mlp_encoder_size"]),
    nn.ReLU(),
    nn.Linear(config["mlp_encoder_size"], config["recurrent_size"]),
    nn.ReLU(),
)
target_pre = copy.deepcopy(pre)
lstm = nn.LSTM(
    input_size=config["recurrent_size"],
    hidden_size=config["recurrent_size"],
    batch_first=True,
)
target_lstm = copy.deepcopy(lstm)
post = nn.Sequential(
    nn.ReLU(),
    nn.Linear(config["recurrent_size"], config["mlp_encoder_size"]),
    nn.ReLU(),
    nn.Linear(config["mlp_encoder_size"], env.action_space.n),
)
nn.init.normal_(post[-1].weight.data, 0, 0.001)
post[-1].bias.data.fill_(0)
target_post = copy.deepcopy(post)

# Modules
state_mod = Seq(
    Mod(pre, in_keys=["observation"], out_keys=["embed"]),
    LSTMModule(lstm=lstm, in_key="embed", out_key="markov_state").set_recurrent_mode(True)
)
target_state_mod = Seq(
    Mod(target_pre, in_keys=["observation"], out_keys=["embed"]),
    LSTMModule(lstm=copy.deepcopy(lstm), in_key="embed", out_key="markov_state").set_recurrent_mode(True),
)
q_mod = Seq(
    Mod(post, in_keys=["markov_state"], out_keys=["action_value"]),
    QValueModule(action_space=env.action_spec),
)
eval_policy = Seq(
    Mod(pre, in_keys=["observation"], out_keys=["embed"]),
    LSTMModule(lstm=lstm, in_key="embed", out_key="markov_state"),
    Mod(post, in_keys=["markov_state"], out_keys=["action_value"]),
    QValueModule(action_space=env.action_spec),
)
policy = EGreedyWrapper(
    eval_policy,
    annealing_num_steps=config["total_frames"],
    spec=env.action_spec,
    eps_init=config["eps_init"],
    eps_end=config["eps_end"],
)
loss_fn = DQNLoss(
    q_mod,
    delay_value=True,
    action_space=env.action_spec,
)
updater = HardUpdate(loss_fn, value_network_update_interval=config["target_delay"])

if hasattr(state_mod[1], "make_tensordict_primer"):
    env.append_transform(state_mod[1].make_tensordict_primer())
    eval_env.append_transform(state_mod[1].make_tensordict_primer())

state_mod(env.reset())

optim = torch.optim.AdamW(
    [*pre.parameters(), *lstm.parameters(), *post.parameters()], lr=config["lr"]
)
collector = SyncDataCollector(
    env,
    policy,
    frames_per_batch=config["segment_length"],
    total_frames=config["total_frames"],
    split_trajs=True,
    init_random_frames=config["random_frames"],
)
eval_collector = SyncDataCollector(
    eval_env,
    eval_policy,
    frames_per_batch=config["eval_length"],
    total_frames=config["total_frames"],
    #split_trajs=True,
    init_random_frames=0,
    reset_at_each_iter=True,
)
rb = TensorDictReplayBuffer(
    storage=LazyTensorStorage(config["buffer_size"]),
    batch_size=config["batch_size"],
    prefetch=config["batch_size"] * config["utd"],
)
pbar = tqdm.tqdm(total=config["epochs"])
longest = 0

best_reward = -float("inf")
eval_reward = -float("inf")
best_eval_reward = -float("inf")
nonzero_frames = 0

wandb.init(project="rdqn_segment", config=config)
for i, data in enumerate(collector):
    frames_this_iter = data[("collector", "mask")].sum()
    nonzero_frames += frames_this_iter
    to_log = {
        "epoch": i,
        "collector/nonzero_frames": nonzero_frames,
    }
    pbar.update()
    # it is important to pass data that is not flattened
    # TODO: data is often longer than this, what does padding do?
    #truncate_trajectories(data, config["segment_length"])
    padded = tensordict.pad(data, [0, 0, 0, config["segment_length"] - data.shape[-1]])
    rb.extend(padded)
    unpadded_data = data.masked_select(data[('collector', 'mask')])
    if collector._frames < config["random_frames"]:
        continue
    mean_reward = unpadded_data["episode_reward"][unpadded_data[("next", "done")]].mean()
    if mean_reward > best_reward:
        best_reward = mean_reward
    for u in range(config["utd"]):
        batch = rb.sample().to(device)
        # Do not use stale states
        # del batch['recurrent_state_c'], batch['recurrent_state_h'], batch[('next', 'recurrent_state_c')], batch[('next', 'recurrent_state_h')]
        state_mod(batch)
        stored_states = batch.select(*[('next', 'recurrent_state_c'), ('next', 'recurrent_state_h')])
        # Prevent leaking learned network states into the target network
        del (
            batch[("next", "recurrent_state_c")],
            batch[("next", "recurrent_state_h")],
        )
        with torch.no_grad():
            target_state_mod(batch["next"])

        # Replace stored states as target will have overwritten them
        batch.update(stored_states)
        # Compute markov_state
        mask = batch[("collector", "mask")]
        masked = batch.masked_select(mask)
        loss = loss_fn(masked)

        loss['loss'].backward()
        optim.step()
        optim.zero_grad()

        #soft_update(target_state_mod, state_mod, config["tau"])
        hard_update(target_state_mod, state_mod, config["target_delay"], i * config["utd"] + u)
        updater.step()

    pbar.set_description(
        f"loss_val: {loss['loss'].item():.5f}, reward(curr, best): ({mean_reward:.2f}, {best_reward:.2f}), eval reward(curr, best): ({eval_reward:.2f}, {best_eval_reward:.2f})"
    )
    if not math.isnan(mean_reward):
        to_log = {**to_log, "train/reward": mean_reward}

    to_log = {
        **to_log,
        "train/loss": loss['loss'].item(),
        "train/best_reward": best_reward,
        "train/epsilon": policy.eps,
        "train/buffer_size": len(rb),
        "collector/action_histogram": wandb.Histogram(data["action"].cpu()),
    }
    policy.step(data.numel())
    collector.update_policy_weights_()

    if i % config["eval_interval"] == 0:
        eval_collector.update_policy_weights_()
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
        }

    wandb.log(to_log)
