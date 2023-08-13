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
from torchrl.objectives.dqn import DQNLoss


class RDQNSegmentLoss(DQNLoss):
    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Computes the DQN loss given a tensordict sampled from the replay buffer.

        This function will also write a "td_error" key that can be used by prioritized replay buffers to assign
            a priority to items in the tensordict.

        Args:
            tensordict (TensorDictBase): a tensordict with keys ["action"] and the in_keys of
                the value network (observations, "done", "reward" in a "next" tensordict).

        Returns:
            a tensor containing the DQN loss.

        """
        td_copy = tensordict.clone(False)
        # for k in self.value_network.recurrent_keys:
        #     del td_copy[k]
        self.value_network(
            td_copy,
            params=self.value_network_params,
        )

        action = tensordict.get(self.tensor_keys.action)
        pred_val = td_copy.get(self.tensor_keys.action_value)

        if self.action_space == "categorical":
            if action.shape != pred_val.shape:
                # unsqueeze the action if it lacks on trailing singleton dim
                action = action.unsqueeze(-1)
            pred_val_index = torch.gather(pred_val, -1, index=action).squeeze(-1)
        else:
            action = action.to(torch.float)
            pred_val_index = (pred_val * action).sum(-1)

        # for k in self.value_network.recurrent_keys:
        #     del td_copy[k]
        td_copy = td_copy.exclude(*self.value_network.recurrent_keys)
        target_value = self.value_estimator.value_estimate(
            td_copy, target_params=self.target_value_network_params
        ).squeeze(-1)

        with torch.no_grad():
            priority_tensor = (pred_val_index - target_value).pow(2)
            priority_tensor = priority_tensor.unsqueeze(-1)
        if tensordict.device is not None:
            priority_tensor = priority_tensor.to(tensordict.device)

        tensordict.set(
            self.tensor_keys.priority,
            priority_tensor,
            inplace=True,
        )
        loss = distance_loss(pred_val_index, target_value, self.loss_function)[tensordict[('collector', 'mask')]]
        return TensorDict({"loss": loss.mean()}, [])


# class RDQNSegmentLoss(LossModule):
#     @dataclass
#     class _AcceptedKeys:
#         """Maintains default values for all configurable tensordict keys.

#         This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
#         default values.

#         Attributes:
#             advantage (NestedKey): The input tensordict key where the advantage is expected.
#                 Will be used for the underlying value estimator. Defaults to ``"advantage"``.
#             value_target (NestedKey): The input tensordict key where the target state value is expected.
#                 Will be used for the underlying value estimator Defaults to ``"value_target"``.
#             value (NestedKey): The input tensordict key where the chosen action value is expected.
#                 Will be used for the underlying value estimator. Defaults to ``"chosen_action_value"``.
#             action_value (NestedKey): The input tensordict key where the action value is expected.
#                 Defaults to ``"action_value"``.
#             action (NestedKey): The input tensordict key where the action is expected.
#                 Defaults to ``"action"``.
#             priority (NestedKey): The input tensordict key where the target priority is written to.
#                 Defaults to ``"td_error"``.
#             reward (NestedKey): The input tensordict key where the reward is expected.
#                 Will be used for the underlying value estimator. Defaults to ``"reward"``.
#             done (NestedKey): The key in the input TensorDict that indicates
#                 whether a trajectory is done. Will be used for the underlying value estimator.
#                 Defaults to ``"done"``.
#         """

#         advantage: NestedKey = "advantage"
#         value_target: NestedKey = "value_target"
#         value: NestedKey = "chosen_action_value"
#         action_value: NestedKey = "action_value"
#         action: NestedKey = "action"
#         priority: NestedKey = "td_error"
#         reward: NestedKey = "reward"
#         done: NestedKey = "done"

#     default_keys = _AcceptedKeys()

#     def __init__(
#         self,
#         memory_module,
#         target_memory_module,
#         q_module,
#         target_q_module,
#         action_space,
#         gamma=0.99,
#     ):
#         super().__init__()

#         self.q_module = q_module
#         self.memory_module = memory_module
#         self.target_q_module = target_q_module
#         self.target_memory_module = target_memory_module
#         self.mask_key = ('collector', 'mask')
#         self.gamma = gamma
#         self.estimator = TD0Estimator(gamma=self.gamma, value_network=None)
#         tensor_keys = {
#             "advantage": self.tensor_keys.advantage,
#             "value_target": self.tensor_keys.value_target,
#             "value": self.tensor_keys.value,
#             "reward": self.tensor_keys.reward,
#             "done": self.tensor_keys.done,
#         }
#         self.estimator.set_keys(**tensor_keys)

#     def forward(self, td):
#         # # Setup the 'next' mask, which is shape [..., b, t]
#         mask = td[self.mask_key]
#         # mem_td = td.select(*self.memory_module.in_keys).exclude(*self.memory_module.recurrent_keys)
#         # next_mem_td = td['next'].select(*self.memory_module.in_keys, strict=False).exclude(*self.memory_module.recurrent_keys)
#         # #next_mask = torch.cat([
#         # #    torch.ones((*td.shape[:-1], 1), device=mask.device, dtype=mask.dtype),
#         # #    mask[..., 1:]
#         # #], dim=-1)
#         # #td[("next", *self.mask_key)] = next_mask


#         # td.set('markov_state', self.memory_module(mem_td.clone()).get("markov_state"))
#         # # Compute target mem first so we don't read from the non-target initial state
#         # with torch.no_grad():
#         #     td.set(('next', 'markov_state'), self.target_memory_module(next_mem_td.clone()).get("markov_state"))

#         # # TODO: Is mask off by one?
#         # # Just roll and slice [1:]
#         # masked = td.masked_select(mask)

#         # masked = self.q_module(masked)
#         # with torch.no_grad():
#         #     masked[("next", "action_value")] = self.target_q_module(masked['next']).get("action_value")
        
#         # breakpoint()
#         # pred_td = masked['chosen_action_value']
#         # td_target = self.estimator.value_estimate(masked)

#         td = td.exclude(*self.memory_module.recurrent_keys)
#         td = self.q_module(self.memory_module(td))

#         with torch.no_grad():
#             td['next'] = td['next'].exclude(*self.memory_module.recurrent_keys)
#             td['next'] = self.target_q_module(self.target_memory_module(td['next']))

#         #pred_td = td['chosen_action_value']
#         #pred_td = torch.gather(td['action_value'], -1, index=td['action']).squeeze(-1)
#         pred_td = torch.sum(td['action_value'] * td['action']).sum(-1)
#         breakpoint()
#         td_target = self.estimator.value_estimate(td)

#         loss = torch.nn.functional.smooth_l1_loss(pred_td[mask], td_target[mask])

#         return TensorDict({"loss": loss}, batch_size=[])
