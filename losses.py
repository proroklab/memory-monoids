import warnings
from dataclasses import dataclass
from typing import Optional, Union

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import dispatch, TensorDictModuleBase
from tensordict.utils import NestedKey
from torch import nn
from torchrl.data.tensor_specs import TensorSpec
from torchrl.modules.tensordict_module.actors import (
    DistributionalQValueActor, QValueActor)
from torchrl.modules.tensordict_module.common import \
    ensure_tensordict_compatible
from torchrl.modules.utils.utils import _find_action_space
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (_GAMMA_LMBDA_DEPREC_WARNING,
                                      ValueEstimators, default_value_kwargs,
                                      distance_loss)
from torchrl.objectives.value import TDLambdaEstimator
from torchrl.objectives.value.advantages import TD0Estimator, TD1Estimator



class RDQNSegmentLoss(LossModule):
    def __init__(
        self,
        memory_module,
        target_memory_module,
        q_module,
        target_q_module,
        action_space,
        gamma=0.99,
    ):
        super().__init__()

        self.q_module = q_module
        self.memory_module = memory_module
        self.target_q_module = target_q_module
        self.target_memory_module = target_memory_module
        self.mask_key = ('collector', 'mask')
        self.gamma = gamma
        self.filter_keys = [
            'action_value',
            'chosen_action_value',
            'done',
            'embed',
            ('collector', 'mask'),
            'reward',
            'observation',
        ]

    def forward(self, td):
        # Setup the 'next' mask, which is shape [..., b, t]
        mask = td[self.mask_key]
        td[("next", *self.mask_key)] = torch.cat([
            torch.ones((*td.shape[:-1], 1), device=mask.device, dtype=mask.dtype),
            mask[..., 1:]
        ], dim=-1)


        # Clone so we don't feed the non-target recurrent states into the target network
        mem_td = td.select(*self.memory_module.in_keys).clone(True)
        mem_td = self.memory_module(mem_td)
        with torch.no_grad():
            next_mem_detach_td = td['next'].select(*self.target_memory_module.in_keys, 'reward', 'done').clone(True)
            next_mem_detach_td = self.target_memory_module(next_mem_detach_td)

        mask = td.get(self.mask_key)
        masked_mem_td = mem_td.masked_select(mask)
        # TODO: Is mask off by one?
        next_masked_mem_detach_td = next_mem_detach_td.masked_select(mask)

        q_td = self.q_module(masked_mem_td)
        with torch.no_grad():
            next_q_detach_td = self.target_q_module(next_masked_mem_detach_td)

        td_target = next_q_detach_td['reward'] + next_q_detach_td['done'].logical_not() * self.gamma * next_q_detach_td['action_value'].max(-1, keepdim=True).values
        pred_td = q_td['chosen_action_value']
        assert td_target.shape == pred_td.shape
        td_error = pred_td - td_target
        loss = td_error.pow(2).mean()

        return TensorDict({"loss": loss}, batch_size=[])