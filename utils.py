import importlib
import math
import gymnasium as gym
from popgym.wrappers import PreviousAction, Antialias, Flatten, DiscreteAction
from torchrl.modules import ConvNet, EGreedyWrapper, LSTMModule, MLP, QValueModule
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import (
    Compose,
    InitTracker,
    StepCounter,
    TransformedEnv,
    RewardSum,
    DoubleToFloat,
)
from torch import nn
import copy
from tensordict.nn import (
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    TensorDictModuleBase,
    EnsembleModule,
)
from linear_attention import LinearAttentionModule



class FinalLinear(nn.Linear):
    def reset_parameters(self):
        nn.init.normal_(self.weight.data, 0, 1e-4)
        self.bias.data.zero_()


def make_pre(config, obs_shape):
    pre = Mod(
        nn.Sequential(
            nn.Linear(obs_shape, config["mlp_encoder_size"]),
            nn.Mish(),
            nn.Linear(config["mlp_encoder_size"], config["recurrent_size"]),
            nn.LayerNorm(config["recurrent_size"], elementwise_affine=False),
            nn.Mish(),
        ),
        in_keys=["observation"],
        out_keys=["embed"],
    )
    return pre

def make_post(config, act_shape):
    post = Mod(
        nn.Sequential(
            nn.Mish(),
            nn.Linear(config["recurrent_size"], config["mlp_encoder_size"]),
            nn.Mish(),
            FinalLinear(config["mlp_encoder_size"], act_shape),
        ),
        in_keys=["markov_state"],
        out_keys=["action_value"]    
    )
    return post

def make_memory(memory_str, config, training):
    if memory_str == "linear_attention":
        mod = Seq(
            LinearAttentionModule(
                in_key="embed",
                out_key="embed",
                input_size=config["recurrent_size"],
                hidden_size=config["recurrent_size"],
                recurrent_keys=["recurrent_state_s0", "recurrent_state_z0"],
            ).temporal_mode(training),
            LinearAttentionModule(
                in_key="embed",
                input_size=config["recurrent_size"],
                hidden_size=config["recurrent_size"],
                recurrent_keys=["recurrent_state_s1", "recurrent_state_z1"],
                out_key="markov_state",
                feed_forward=False,
            ).temporal_mode(training),
        )
        mod.recurrent_keys = ["recurrent_state_s0", "recurrent_state_z0", "recurrent_state_s1", "recurrent_state_z1"]
    elif memory_str == "lstm":
        mod = LSTMModule(
            input_size=config["recurrent_size"],
            hidden_size=config["recurrent_size"],
            in_key="embed",
            out_key="markov_state"
        ).set_recurrent_mode(training)
        mod.recurrent_keys = ["recurrent_state_c", "recurrent_state_h"]
    elif memory_str == "none":
        mod = Mod(nn.Identity(), in_keys=["embed"], out_keys=["markov_state"])
        mod.recurrent_keys = []
    else:
        raise NotImplementedError()
    
    return mod

def get_modules(config, env, training):
    device = config["device"]
    obs_shape = math.prod(env.observation_spec["observation"].shape)
    act_shape = env.action_space.n
    pre = make_pre(config, obs_shape)
    memory = make_memory(config["memory_name"], config, training)
    post = make_post(config, act_shape)

    module = Seq(
        pre,
        memory,
        post,
        QValueModule(action_space=env.action_spec),
    ).to(device)
    module.recurrent_keys = memory.recurrent_keys
    # Important: init parameters before initializing optimizer
    # otherwise lstm might not have any params
    module(env.reset())
    return module

    # target_module = Seq(
    #     target_pre,
    #     target_memory,
    #     target_post
    # )

    # state_mod = Seq(
    #     pre,
    #     memory
    # )
    # state_mod.recurrent_keys = memory.recurrent_keys

    # target_state_mod = Seq(
    #     target_pre,
    #     target_memory
    # )
    # target_state_mod.recurrent_keys = memory.recurrent_keys

    # q_net = Seq(
    #     post,
    #     QValueModule(action_space=env.action_spec),
    # )
    # target_q_net = Seq(
    #     target_post,
    #     QValueModule(action_space=env.action_spec)
    # )

    # for p in target_state_mod.parameters():
    #     p.requires_grad_(False)
    # for p in target_q_net.parameters():
    #     p.requires_grad_(False)

    # return state_mod, q_net, target_state_mod, target_q_net


def apply_primers(env, modules):
    try:
        for m in modules:
            primer = m.make_tensordict_primer()
            env.append_transform(primer)
    except TypeError:
        if hasattr(modules, "make_tensordict_primer"):
            primer = modules.make_tensordict_primer()
            env.append_transform(primer)

    return env
    # try:
    #     for m in module:
    #         print(m)
    #         return apply_primers(env, m)
    # except TypeError:
    #     pass
    # if isinstance(module, TensorDictModuleBase) and hasattr(module, "make_tensordict_primer"):
    #     primer = m.make_tensordict_primer()
    #     env.append_transform(primer)
    # return env


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source, delay, update):
    if update % delay == 0:
        target.load_state_dict(source.state_dict())

def load_popgym_env(config, eval=False):
    module, cls = config["env"].rsplit(".", 1)
    mod = importlib.import_module(module)
    instance = getattr(mod, cls)()
    instance = Flatten(Antialias(PreviousAction(instance)))
    if isinstance(instance.action_space, gym.spaces.MultiDiscrete):
        instance = DiscreteAction(instance)
    instance.action_space.seed(config["seed"] + eval * 1000)

    return instance

def load_wrapped_popgym_env(config, device, eval=False):
    instance = load_popgym_env(config, eval=eval)
    env = TransformedEnv(
        GymWrapper(instance, device=device),
        Compose(
            StepCounter(),
            InitTracker(),
            RewardSum(),
            DoubleToFloat(in_keys=["observation"]),
        ),
    )
    # TODO: torchrl cannot seed gymnasium-specced envs
    # env.seed(config["seed"] + eval * 1000)

    return env


def truncate_trajectories(td, segment_length):
    segment_length = 10
    lengths = td[('collector', 'mask')].sum(dim=1)
    segment_index_repeats = torch.ceil(lengths.float() / segment_length).int()
    num_segments = torch.sum(segment_index_repeats)
    batch_index = []
    time_index = []
    offset = 0
    for length, segment_index_repeat in zip(lengths, segment_index_repeats):
        #t_index = torch.arange(length).repeat(segment_index_repeat.item())
        index_index = torch.arange(length)
        b_index = torch.arange(segment_index_repeat).repeat_interleave(segment_length)[index_index] + offset
        offset += segment_index_repeat
        t_index = torch.arange(segment_length).repeat(segment_index_repeat.item())[index_index]
        batch_index.append(b_index)
        time_index.append(t_index)

    batch_index = torch.cat(batch_index)
    time_index = torch.cat(time_index)

    breakpoint()
    #unpadded_segment_lens = 
    batch_index = torch.arange(num_segments)
    breakpoint()
    batch_index = torch.repeat_interleave(torch.arange(num_segments), truncated_lengths)
    time_index = torch.arange(segment_length).repeat(num_segments)
    indices = torch.stack([batch_index, time_index], dim=0)

    for k, v in list(tensordict.items()):
        td[k] = torch.zeros(num_segments, segment_length, *v.shape[2:], dtype=v.dtype, device=v.device).scatter_(
            dim=0, index=indices, src=v,
        )

    return td
