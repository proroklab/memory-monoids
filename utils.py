import importlib
import gymnasium as gym
from popgym.wrappers import PreviousAction, Antialias, Flatten, DiscreteAction
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import (
    Compose,
    InitTracker,
    StepCounter,
    TransformedEnv,
    RewardSum,
    DoubleToFloat,
)


def apply_primers(env, modules):
    for m in modules:
        if hasattr(m, "make_tensordict_primer"):
            primers = m.make_tensordict_primer()
            if not isinstance(primers, (list, tuple)):
                primers = [primers]
            for p in primers:
                env.append_transform(p)
    return env


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
