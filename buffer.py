from typing import Any, Callable, Optional, Tuple, Union
from torchrl.buffer import ReplayBuffer, Storage, Writer, Sampler, PrioritizedSampler
import torch
import numpy as np


class StreamingSampler(Sampler):
    def __init__(
        self,
        seek_to_episode_start: bool = True,
        discard_trailing_episode: bool = False,
    ):
        self.seek_to_episode_start = seek_to_episode_start
        self.discard_trailing_episode = discard_trailing_episode
        self.episode_start_idx = torch.tensor([], dtype=torch.int64)

    def sample(self, storage: Storage, batch_size: int) -> Tuple[torch.Tensor, dict]:
        if len(storage) == 0:
            raise RuntimeError("Storage is empty")

        sequence_len = min(batch_size, len(storage))
        if self.seek_to_episode_start:
            available_starts = len(storage) - self.episode_start_idx >= sequence_len 
            start_idx = torch.randint(0, available_starts.numel())
            start = available_starts[start_idx]
        else:
            start = torch.randint(0, len(storage) - sequence_len)
        index = torch.arange(start, start + sequence_len) 
        return index, {}

    def update_episode_boundaries(self, start_index):
        self.episode_start_idx = torch.cat([self.episode_start_idx, start_index], dim=0)


class PrioritizedStreamingSampler(PrioritizedSampler):
    def __init__(
        self,
        max_capacity: int,
        alpha: float,
        beta: float,
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float,
        reduction: str = "max",
        seek_to_episode_start: bool = True,
    ) -> None:
        super().__init__(
            max_capacity=max_capacity,
            alpha=alpha,
            beta=beta,
            eps=eps,
            dtype=dtype,
            reduction=reduction,
        )
        self.seek_to_episode_start = seek_to_episode_start
        self.episode_start_idx = torch.tensor([], dtype=torch.int64)

    def sample(self, storage: Storage, batch_size: int) -> torch.Tensor:
        if len(storage) == 0:
            raise RuntimeError("Empty storage")
        p_sum = self._sum_tree.query(0, len(self.episode_start_idx))
        p_min = self._min_tree.query(0, len(self.episode_start_idx))
        if p_sum <= 0:
            raise RuntimeError("negative p_sum")
        if p_min <= 0:
            raise RuntimeError("negative p_min")
        mass = np.random.uniform(0.0, p_sum, size=batch_size)
        index = self._sum_tree.scan_lower_bound(mass)
        if not isinstance(index, np.ndarray):
            index = np.array([index])

        #if isinstance(index, torch.Tensor):
        #    index.clamp_max_(len(storage) - 1)
        #else:
        #    index = np.clip(index, None, len(storage) - 1)
        weight = self._sum_tree[index]

        # Importance sampling weight formula:
        #   w_i = (p_i / sum(p) * N) ^ (-beta)
        #   weight_i = w_i / max(w)
        #   weight_i = (p_i / sum(p) * N) ^ (-beta) /
        #       ((min(p) / sum(p) * N) ^ (-beta))
        #   weight_i = ((p_i / sum(p) * N) / (min(p) / sum(p) * N)) ^ (-beta)
        #   weight_i = (p_i / min(p)) ^ (-beta)
        # weight = np.power(weight / (p_min + self._eps), -self._beta)
        weight = np.power(weight / p_min, -self._beta)

        if self.seek_to_episode_start:
            available_starts = len(storage) - self.episode_start_idx >= batch_size 
            start_idx = torch.randint(0, available_starts.numel())
            start = available_starts[start_idx]
        else:
            start = index
        indices = torch.arange(start, start + batch_size)
        return indices, {"_weight": weight}

    def update_episode_boundaries(self, start_index):
        self.episode_start_idx = torch.cat([self.episode_start_idx, start_index], dim=0)