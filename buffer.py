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
        #shuffle_interval=200,
        swap_iters=1,
    ):
        super().__init__(buffer_size, schema)
        self.start_key = start_key
        self.seek_to_start = seek_to_start
        self.swap_iters = swap_iters
        #self.shuffle_interval = shuffle_interval
        self.transition_counter = 0
        if self.seek_to_start:
            self.episode_starts = deque()
        #if shuffle_interval is not None:
        #    self.last_shuffle_at = 0
        assert self.data[start_key].ndim == 1

    def sample(self, size: int, key: jax.random.PRNGKey) -> Dict[str, np.ndarray]:
        out = {}
        # TODO: Should not sample between ptr and the next index
        # This region is guaranteed to be corrupted
        # start_idx = random.choice(self.episode_starts)
        assert self.size >= size, f"Buffer size {self.size} is less than sample size {size}"
        # if self.shuffle_interval is not None and self.last_shuffle_at <= self.shuffle_interval:
        #     print("Shuffling buffer")
        #     self.shuffle(key)
        #     self.last_shuffle_at = self.transition_counter

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

    def sample_noncontiguous(self, size: int, key: jax.random.PRNGKey) -> Dict[str, np.ndarray]:
        out = {}
        assert self.size >= size, f"Buffer size {self.size} is less than sample size {size}"
        rng = np.random.default_rng(jax.random.bits(key).item())
        sample_idxs = []
        while len(sample_idxs) < size:
            start_idx = rng.integers(len(self.episode_starts) - 1 - 1)
            start = self.episode_starts[start_idx]
            end = self.episode_starts[start_idx + 1]
            sample_idxs.append(
                np.arange(start, end)
            )

        sample_idxs = np.concatenate(sample_idxs)[:size]
        # sample_idxs = np.r_[*sample_slices]
        for k, v in self.data.items():
            out[k] = v[sample_idxs]
        return out

    def swap(self, key) -> None:
        # Shuffle two consecutive elements
        # Don't shuffle the very last element or the second to last element
        if len(self.episode_starts) < 4:
            return

        rng = np.random.default_rng(jax.random.bits(key).item())
        if self.swap_iters == "auto":
            r = range(int(np.log2(self.size)))
        else:
            r = range(self.swap_iters)
        for _ in r:
            idx = rng.integers(len(self.episode_starts) - 1 - 2)
            idx_a, idx_b, idx_c = self.episode_starts[idx], self.episode_starts[idx + 1], self.episode_starts[idx + 2]
            idxs_a = np.arange(idx_a, idx_b)
            idxs_b = np.arange(idx_b, idx_c)

            src_idx = jnp.concatenate([idxs_a, idxs_b])
            sink_idx = jnp.concatenate([idxs_b, idxs_a])
            for k in self.data:
                self.data[k][src_idx] = self.data[k][sink_idx]
            
            #assert self.data[self.start_key][idx_a] == True
            #assert self.data['next_done'][idx_c - 1] == True

            #self.episode_starts[idx] = idx_b - idx_a
            self.episode_starts[idx + 1] = idx_c - idx_b + idx_a

        
    def shuffle(self, key) -> None:
        # Shuffle all elements except for the last one,
        # as it could be a partial fragment
        rng = np.random.default_rng(jax.random.bits(key).item())
        starts_to_shuffle = list(self.episode_starts)[:-1]
        frag_starts, frag_ends = starts_to_shuffle[:-1], starts_to_shuffle[1:]
        boundaries = np.array([frag_starts, frag_ends])
        rng.shuffle(boundaries, axis=1)
        starts, ends = boundaries[0], boundaries[1]
        lens = ends - starts
        breakpoint()
        shuffled_idx = np.repeat(ends - lens.cumsum(), 1) + np.arange(lens.sum())

        for k in self.data:
            #shuffled_data = np.take_along_axis(shuffled_idx, 0, self.data[k][self.episode_starts[:-1]])
            shuffled_data = np.take_along_axis(shuffled_idx, 0, self.data[k][starts_to_shuffle])
            #data = np.concatenate([shuffled_data, self.data[self.episode_starts[:-1]:]])
            data = np.concatenate([shuffled_data, self.data[starts_to_shuffle[-1]:]])
            self.data[k] = data
        
        # Finally set shuffled index
        self.episode_starts = deque(starts + ends[-1:] + self.episode_starts[:-1])

    def add(self, key, **data) -> None:
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
        self.transition_counter += batch_size
        self.size = min(self.transition_counter, self.max_size)

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

