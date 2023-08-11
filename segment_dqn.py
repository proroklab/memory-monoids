import copy
import math
import torch
import tqdm
from linear_attention import LinearAttentionModule
import popgym
from utils import hard_update, soft_update, load_popgym_env, load_wrapped_popgym_env, truncate_trajectories
import numpy as np
import gymnasium as gym
import importlib
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq, EnsembleModule
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


class FinalLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0, 1e-4)
        self.bias.data.zero_()
        

a = argparse.ArgumentParser()
a.add_argument("config", type=str)
a.add_argument("--seed", '-s', type=int, default=0)
a.add_argument("--device", '-d', type=str, default='cpu')
args = a.parse_args()

with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config['seed'] = args.seed
config['device'] = args.device
#config["epochs"] = config["total_frames"] // config["segment_length"]
config["total_frames"] = config["epochs"] * config["segment_length"]
config["total_random_frames"] = config["random_epochs"] * config["segment_length"]

torch.manual_seed(config["seed"])

device = torch.device(config['device']) 
env = load_wrapped_popgym_env(config, device)
eval_env = load_wrapped_popgym_env(config, device, eval=True)
obs_shape = math.prod(env.observation_spec['observation'].shape)
act_shape = env.action_space.n

# Networks
pre = nn.Sequential(
    nn.Linear(obs_shape, config["mlp_encoder_size"]),
    nn.Mish(),
    nn.Linear(config["mlp_encoder_size"], config["recurrent_size"]),
    nn.LayerNorm(config["recurrent_size"], elementwise_affine=False),
    nn.Mish(),
).to(device)
target_pre = copy.deepcopy(pre).to(device)
# lstm = nn.LSTM(
#     input_size=config["recurrent_size"],
#     hidden_size=config["recurrent_size"],
#     batch_first=True,
# ).to(device)
#target_lstm = copy.deepcopy(lstm).to(device)
post = nn.Sequential(
    nn.Mish(),
    nn.Linear(config["recurrent_size"], config["mlp_encoder_size"]),
    nn.Mish(),
    FinalLinear(config["mlp_encoder_size"], act_shape),
).to(device)

# Modules
state_mod = Seq(
    Mod(pre, in_keys=["observation"], out_keys=["embed"]),
    #LSTMModule(lstm=lstm, in_key="embed", out_key="markov_state").set_recurrent_mode(True)
    LinearAttentionModule(in_key="embed", out_key="markov_state", input_size=config["recurrent_size"], hidden_size=config["recurrent_size"]),
)
lstm = state_mod[1].module
target_state_mod = Seq(
    Mod(target_pre, in_keys=["observation"], out_keys=["embed"]),
    #LSTMModule(lstm=copy.deepcopy(lstm), in_key="embed", out_key="markov_state").set_recurrent_mode(True),
    LinearAttentionModule(in_key="embed", out_key="markov_state", input_size=config["recurrent_size"], hidden_size=config["recurrent_size"]),
)
q_mod = Seq(
    Mod(post, in_keys=["markov_state"], out_keys=["action_value"]),
    QValueModule(action_space=env.action_spec),
)
eval_policy = Seq(
    Mod(pre, in_keys=["observation"], out_keys=["embed"]),
    #LSTMModule(lstm=lstm, in_key="embed", out_key="markov_state"),
    LinearAttentionModule(in_key="embed", out_key="markov_state", input_size=config["recurrent_size"], hidden_size=config["recurrent_size"]),
    Mod(post, in_keys=["markov_state"], out_keys=["action_value"]),
    QValueModule(action_space=env.action_spec),
)
policy = EGreedyWrapper(
    eval_policy,
    annealing_num_steps=config["epochs"],
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
    init_random_frames=config["total_random_frames"],
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
pbar = tqdm.tqdm(total=config["epochs"] + config["random_epochs"])
longest = 0

best_reward = -float("inf")
eval_reward = -float("inf")
best_eval_reward = -float("inf")
nonzero_frames = 0

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
    #truncate_trajectories(data, config["segment_length"])
    padded = tensordict.pad(data, [0, 0, 0, config["segment_length"] - data.shape[-1]])
    rb.extend(padded.cpu())
    unpadded_data = data.masked_select(data[('collector', 'mask')])
    if collector._frames < config["total_random_frames"]:
        continue
    mean_reward = unpadded_data["episode_reward"][unpadded_data[("next", "done")]].mean()
    if mean_reward > best_reward:
        best_reward = mean_reward
    for u in range(config["utd"]):
        if i % config["reset_interval"] == 0:
            lstm.reset_parameters()
            pre.apply(lambda x: x.reset_parameters() if hasattr(x, "reset_parameters") else None)
            post.apply(lambda x: x.reset_parameters() if hasattr(x, "reset_parameters") else None)
            
        batch = rb.sample().to(device)
        # Do not use stale states
        # del batch['recurrent_state_c'], batch['recurrent_state_h'], batch[('next', 'recurrent_state_c')], batch[('next', 'recurrent_state_h')]
        state_mod(batch)
        # Prevent leaking learned network states into the target network
        # Otherwise s_0 from learned net will feed into target network at initial timestep
        # del (
        #     batch[("next", "recurrent_state_s")],
        #     batch[("next", "recurrent_state_z")],
        # )
        with torch.no_grad():
            target_state_mod(batch["next"])

        # Replace stored states as target will have overwritten them
        #batch.update(stored_states)
        # Compute markov_state
        mask = batch[("collector", "mask")]
        masked = batch.masked_select(mask)
        loss = loss_fn(masked)

        loss['loss'].backward()
        optim.step()
        optim.zero_grad()

        #soft_update(target_state_mod, state_mod, config["tau"])
        hard_update(
            target_state_mod, 
            state_mod, 
            config["target_delay"], 
            i * config["utd"] + u
        )
        updater.step()

    pbar.set_description(
        f"loss: {loss['loss'].item():.3f}, rew(c:{mean_reward:.2f}, b:{best_reward:.2f}), eval rew: (c:{eval_reward:.2f}, b:{best_eval_reward:.2f})"
    )
    if not math.isnan(mean_reward):
        to_log = {**to_log, "train/reward": mean_reward}

    to_log = {
        **to_log,
        "train/loss": loss['loss'].item(),
        "train/best_reward": best_reward,
        "train/epsilon": policy.eps,
        "train/buffer_size": len(rb),
        "train/buffer_capacity": len(rb) / config["buffer_size"],
        "collector/action_histogram": wandb.Histogram(data["action"].cpu()),
    }
    policy.step(1)
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
