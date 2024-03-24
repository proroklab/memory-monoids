from typing import Dict, Tuple
import numpy as np
import jax
from collections import deque


class ReplayBufferBase:
    """Base class for replay buffer"""

    def __init__(self, buffer_size: int, schema: Dict[str, np.shape], preallocate_memory=True):
        self.data: Dict[str, np.ndarray] = {}
        self.dtypes: Dict[str, np.dtype] = {}
        self.ptr = 0
        self.size = 0
        self.transitions_added = 0
        self.max_size = buffer_size
        for k, v in schema.items():
            shape = v["shape"]
            if isinstance(shape, int):
                shape = (shape,)
            self.data[k] = np.zeros((buffer_size, *shape), dtype=v["dtype"])
            if preallocate_memory:
                # Allocate memory all at once, so we get OOM errors early
                self.data[k].fill(0)

    def __len__(self):
        return self.size

    def get_stored_size(self):
        return self.size

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

    def on_episode_end(self):
        pass

    def get_density(self):
        return self.density if hasattr(self, 'density') else -1.0


class ReplayBuffer(ReplayBufferBase):
    """A standard replay buffer using uniform sampling. This
    may be used to implement a segment-based buffer."""
    def __init__(self, buffer_size: int, schema: Dict[str, np.shape]):
        super().__init__(buffer_size, schema)
        self.unpadded_size = 0

    def sample(self, size: int, key: jax.random.PRNGKey) -> Dict[str, np.ndarray]:
        out = {}
        rng = np.random.default_rng(jax.random.bits(key).item())
        idx = rng.integers(size=(size,), low=0, high=self.size)
        for k, v in self.data.items():
            out[k] = v[idx]
        return out

    def add(self, **data) -> None:
        data, batch_size = self.validate_inputs(data)
        idx = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
        self.transitions_added += batch_size

        if 'mask' in data:
            self.unpadded_size += data['mask'].sum()
            self.density = self.unpadded_size / max(1, self.transitions_added)

        for k in self.data:
            assert k in data

        for k, v in data.items():
            self.data[k][idx] = np.array(v, copy=True)




class TapeBuffer(ReplayBufferBase):
    def __init__(
        self,
        buffer_size: int,
        start_key: str,
        schema: Dict[str, np.shape],
    ):
        super().__init__(buffer_size, schema)
        self.start_key = start_key
        self.episode_starts = deque()
        self.valid_transitions = 0
        assert self.data[start_key].ndim == 1

    def sample(self, size: int, key: jax.random.PRNGKey) -> Dict[str, np.ndarray]:
        out = {k: [] for k in self.data}
        rng = np.random.default_rng(jax.random.bits(key).item())
        count = 0
        while count < size:
            start_idx = rng.integers(len(self.episode_starts) - 1 - 1)
            start = self.episode_starts[start_idx]
            end = self.episode_starts[start_idx + 1]
            # Special case if we wrap around
            # e.g., [3, 6, 10, 12, 4] (we select 12)
            if end < start:
                count += self.max_size - start + end
                for k, v in self.data.items():
                    out[k].append(v[start:])
                    out[k].append(v[:end])
            else:
                count += end - start
                for k, v in self.data.items():
                    out[k].append(v[start:end])

        return {k: np.concatenate(v)[:size] for k, v in out.items()}

    def get_num_transitions(self, idx_a, idx_b):
        """Return the number of transitions between idx_a and idx_b.

        idx_a might wrap around max_len before getting to idx b.
        """
        if idx_a < idx_b:
            # Standard case
            return idx_b - idx_a
        else:
            return self.max_size - idx_a + idx_b


    def add_simple(self, **data) -> None:
        """This is a simpler version of add.
        
        We do not use it as it has O(B) time complexity where B
        is the length of the input (not the buffer, but the new data).
        However, it is easier to implement and understand. Feel free to use
        this instead of add
        """
        data, batch_size = self.validate_inputs(data)

        # Add new start indices to episode_starts
        idx = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        new_starts = self.ptr + np.flatnonzero(data[self.start_key])
        self.ptr = (self.ptr + batch_size) % self.max_size

        # Pop all starts in the new indices
        # We will be overwriting these episodes
        # O(batch_size) time complexity
        while self.episode_starts and self.episode_starts[0] in idx: 
            self.episode_starts.popleft()

        # Add new indices
        self.episode_starts.extend(new_starts.tolist())
        self.transitions_added += batch_size
        self.size = min(self.max_size, self.size + batch_size)

        for k, v in data.items():
            self.data[k][idx] = np.array(v, copy=True)

        assert self.size <= self.max_size

    def add(self, **data) -> None:
        """This function differs slightly from that in the paper. Rather than
        append and pop from the left, which is O(n) for an array, we simulate 
        a circular buffer using in-place editing of the array. This results in O(1)
        "popping" from the left. 
        
        We maintain a pointer which tells us our position in the circular buffer. As
        data moves out of the circular buffer, we delete their indices from episode_starts.
        This prevents us from sampling this data. The old data will eventually be
        overwritten. 
        """
        data, batch_size = self.validate_inputs(data)

        # Buffer full
        # find episodes that we are going to overwrite
        # and remove their start indices from episode_starts
        # Pop until enough free space
        while (self.size + batch_size) > self.max_size:
            popped = self.episode_starts.popleft()
            # TODO: Handle case where indices wrap around, e.g. [8, 0, 2, 4] case
            if popped > self.episode_starts[0]:
                num_transitions_popped = self.max_size - popped + self.episode_starts[0]
            # Normal case, e.g. [3, 4, 9, 1]
            else:
                num_transitions_popped = self.episode_starts[0] - popped
            self.size -= num_transitions_popped


        # Add new start indices to episode_starts
        idx = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        new_starts = self.ptr + np.flatnonzero(data[self.start_key])
        self.episode_starts.extend(new_starts.tolist())

        # Move the pointer, wrap around the array if necessary. This is required
        # as we update the array in-place instead of append/popleft
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.transitions_added += batch_size
        self.size += batch_size

        for k, v in data.items():
            self.data[k][idx] = np.array(v, copy=True)

        assert self.size <= self.max_size



def test_add_sample():
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

    for key in keys:
        sam = b.sample(2, key)
        assert sam['start'][0] == True

def test_wraparound():
    b = TapeBuffer(
        5,
        "start",
        {
            "a": {"shape": (), "dtype": np.int32},
            "start": {"shape": (), "dtype": bool},
        },
    )
   
    data0 = {
        "a": np.arange(2),
        "start": np.array([True, False])
    }
    data1 = {
        "a": 2 + np.arange(2),
        "start": np.array([True, False])
    }
    data2 = {
        "a": 4 + np.arange(2),
        "start": np.array([True, False])
    }
    b.add(**data0)
    b.add(**data1)
    b.add(**data2)
    # Goes from [0, 1, 2, 3, 4, 5] -> [5, 1, 2, 3, 4]
    #           [s, _, s, _, s, _] -> [s, _, s, _, s] 
    expected_a = np.array([5, 1, 2, 3, 4])
    expected_start = np.array([0, 0, 1, 0, 1])
    expected_size = 4

    # Check data is correct
    assert np.all(expected_a == b.data['a']), f"expected: {expected_a}\nactual: {b.data['a']}"
    # Check flags 
    assert np.all(expected_start == b.data['start']), f"expected: {expected_start}\nactual: {b.data['start']}"
    # Check indices
    assert np.all(np.where(expected_start)[0] == b.episode_starts), f"expected: {np.where(expected_start)[0]}\nactual: {b.episode_starts}"
    # Check size
    # This test fails when using simple as size is always max_size once we reach max_size
    # Don't worry about it
    assert b.size == expected_size, f"expected: {expected_size}\nactual {b.size}"

def test_wraparound2():
    b = TapeBuffer(
        10,
        "start",
        {
            "a": {"shape": (), "dtype": np.int32},
            "start": {"shape": (), "dtype": bool},
        },
    )

    # [0, 1, ..., 38, 39]
    datas = [
        {
            "a": i * 2 + np.arange(2),
            "start": np.array([1, 0])
        }
        for i in range(20)
    ]
    for d in datas:
        b.add(**d)

    # Goes from [0, 1, 2, 3, 4, 5] -> [5, 1, 2, 3, 4]
    #           [s, _, s, _, s, _] -> [s, _, s, _, s] 
    expected_a = np.array([30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
    expected_start = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    expected_size = 10

    # Check data is correct
    assert np.all(expected_a == b.data['a']), f"expected: {expected_a}\nactual: {b.data['a']}"
    # Check flags 
    assert np.all(expected_start == b.data['start']), f"expected: {expected_start}\nactual: {b.data['start']}"
    # Check indices
    assert np.all(np.where(expected_start)[0] == b.episode_starts), f"expected: {np.where(expected_start)[0]}\nactual: {b.episode_starts}"
    # Check size
    assert b.size == expected_size, f"expected: {expected_size}\nactual {b.size}"

if __name__ == "__main__":
    test_add_sample()
    test_wraparound()
    test_wraparound2()