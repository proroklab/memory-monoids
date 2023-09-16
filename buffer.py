from typing import Dict, Tuple
import numpy as np
import jax
import jax.numpy as jnp
import random
from collections import deque


class ReplayBuffer:
    """A standard replay buffer using uniform sampling. This
    may be used to implement a segment-based buffer."""

    def __init__(self, buffer_size: int, schema: Dict[str, np.shape], contiguous=False):
        self.data: Dict[str, np.ndarray] = {}
        self.dtypes: Dict[str, np.dtype] = {}
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size
        self.padding_size = 0
        self.contiguous = contiguous
        for k, v in schema.items():
            shape = v["shape"]
            if isinstance(shape, int):
                shape = (shape,)
            self.data[k] = np.zeros((buffer_size, *shape), dtype=v["dtype"])

    def __len__(self):
        return self.size

    def sample(self, size: int, key: jax.random.PRNGKey) -> Dict[str, np.ndarray]:
        out = {}
        rng = np.random.default_rng(jax.random.bits(key).item())
        if self.contiguous:
            #start_idx = jax.random.randint(key, shape=(), minval=0, maxval=self.size)
            start_idx = rng.integers(size=(), low=0, high=self.size)
            idx = np.arange(start_idx, start_idx + size) % self.size
        else:
            #idx = jax.random.randint(key, shape=(size,), minval=0, maxval=self.size)
            idx = rng.integers(size=(size,), low=0, high=self.size)
        for k, v in self.data.items():
            out[k] = v[idx]
        return out

    def validate_inputs(
        self, data: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], int]:
        for k in data:
            assert k in self.data, f"Key {k} not in schema"
            # Allow the user to omit the batch dim (pass a single transition)
            if data[k].ndim == self.data[k].ndim - 1:
                data[k] = np.expand_dims(data[k], 0)

            if data[k].ndim > 1 or self.data[k].ndim > 1:
                assert data[k].shape[1:] == self.data[k].shape[1:]

        batch_sizes = [d.shape[0] for d in data.values()]
        assert all(batch_sizes[0] == s for s in batch_sizes)
        batch_size = batch_sizes[0]
        return data, batch_size

    def add(self, **data) -> None:
        if 'mask' in data:
            self.padding_size += data['mask'].sum()
            self.density = (self.size - self.padding_size) / max(1, self.size)
        data, batch_size = self.validate_inputs(data)
        idx = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

        for k in self.data:
            assert k in data

        for k, v in data.items():
            self.data[k][idx] = np.array(v, copy=False)

    def on_episode_end(self):
        pass

    def get_stored_size(self):
        return self.size

    def get_density(self):
        return self.density if hasattr(self, 'density') else -1.0


class TapeBuffer(ReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        start_key: str,
        schema: Dict[str, np.shape],
        seek_to_start=True,
    ):
        super().__init__(buffer_size, schema)
        self.start_key = start_key
        self.seek_to_start = seek_to_start
        if self.seek_to_start:
            self.episode_starts = deque()
        assert self.data[start_key].ndim == 1

    def sample(self, size: int, key: jax.random.PRNGKey) -> Dict[str, np.ndarray]:
        out = {}
        # TODO: Should not sample between ptr and the next index
        # This region is guaranteed to be corrupted
        # start_idx = random.choice(self.episode_starts)
        assert self.size >= size, f"Buffer size {self.size} is less than sample size {size}"
        rng = np.random.default_rng(jax.random.bits(key).item())
        if self.seek_to_start:
            #start_idx = jax.random.choice(key, np.array(self.episode_starts))
            start_idx = rng.choice(np.array(self.episode_starts))
        else:
            #start_idx = jax.random.choice(key, self.size)
            start_idx = rng.randint(self.size)
        idx = np.arange(start_idx, start_idx + size) % self.size
        for k, v in self.data.items():
            out[k] = v[idx]
        return out

    def add(self, **data) -> None:
        data, batch_size = self.validate_inputs(data)

        idx = np.arange(self.ptr, self.ptr + batch_size) % self.max_size

        if self.seek_to_start:
            # TODO: We need to zero the data after the episode starts?
            # Or what happens? We will replay a partial episode
            # At most 1 partial episode will exist, can we ignore it?

            # Find starts that we are going to overwrite
            # and remove them from the list
            if len(self.episode_starts) > 0 and (self.size + batch_size) >= self.max_size: 
                while True:
                    if self.episode_starts[0] >= self.ptr and self.episode_starts[0] < self.ptr + batch_size:
                        self.episode_starts.popleft()
                    else:
                        break

            new_starts = self.ptr + np.flatnonzero(data[self.start_key])
            self.episode_starts.extend(new_starts.tolist())

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

        for k, v in data.items():
            self.data[k][idx] = np.array(v, copy=False)


if __name__ == "__main__":
    b = TapeBuffer(
        10,
        "start",
        {
            "a": {"shape": (2, 3), "dtype": np.int32},
            "b": {"shape": (3, 4), "dtype": np.float32},
            "start": {"shape": (), "dtype": bool},
        },
    )
    start0 = np.array([True, False, False, True, False])
    data = {
        "a": np.arange(5 * 2 * 3).reshape((5, 2, 3)),
        "b": np.arange(5 * 3 * 4).reshape((5, 3, 4)),
        "start": start0,
    }
    b.add(**data)

    start1 = np.array([True, False])
    data2 = {
        "a": data["a"].max() + 1 + np.arange(2 * 2 * 3).reshape((2, 2, 3)),
        "b": data["b"].max() + 1 + np.arange(2 * 3 * 4).reshape((2, 3, 4)),
        "start": start1,
    }
    b.add(**data2)
    assert (np.concatenate([data['a'], data2['a']]) == b.data['a'][:7]).all()
    assert (np.concatenate([data['b'], data2['b']]) == b.data['b'][:7]).all()
    assert (np.concatenate([data['start'], data2['start']]) == b.data['start'][:7]).all()
    keys = jax.random.split(jax.random.PRNGKey(0), 100)

    around = b.sample(7, keys[0])

    for key in keys:
        sam = b.sample(2, key)
        assert sam['start'][0] == True

