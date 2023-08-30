from typing import Dict
import numpy as np
import random


class ReplayBuffer:
    """A standard replay buffer using uniform sampling. This
    may be used to implement a segment-based buffer."""
    def __init__(self, buffer_size: int, schema: Dict[str, np.shape]):
        self.data: Dict[str, np.ndarray] = {}
        self.dtypes: Dict[str, np.dtype] = {}
        self.ptr = 0
        self.size = 0
        self.max_size = buffer_size
        for k, v in schema.items():
            shape = v['shape']
            if isinstance(shape, int):
                shape = (shape,)
            self.data[k] = np.zeros((buffer_size, *shape), dtype=v['dtype'])

    def __len__(self):
        return self.size
    
    def sample(self, size: int) -> Dict[str, np.ndarray]:
        out = {}
        idx = np.random.randint(low=0, high=self.size, size=size)
        for k, v in self.data.items():
            out[k] = v[idx]
        return out

    #def add(self, data: Dict[str, np.ndarray]) -> None:
    def add(self, **data) -> None:
        for k, v in data.items():
            assert k in self.data
            if v.ndim > 1 or self.data[k].ndim > 1:
                assert v.shape[1:] == self.data[k].shape[1:]

        batch_sizes = [d.shape[0] for d in data.values()]
        assert all(batch_sizes[0] == s for s in batch_sizes)
        batch_size = batch_sizes[0]

        idx = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

        for k in self.data:
            assert k in data

        for k, v in data.items():
            self.data[k][idx] = np.array(v, copy=False)


class TapeBuffer(ReplayBuffer):
    def __init__(self, buffer_size: int, start_key: str, schema: Dict[str, np.shape]):
        super().__init__(buffer_size, schema)
        self.episode_starts = []
        self.start_key = start_key
        assert self.data[start_key].ndim == 1

    def sample(self, size: int) -> Dict[str, np.ndarray]:
        out = {}
        # TODO: Should not sample between ptr and the next index
        # This region is guaranteed to be corrupted
        start_idx = random.choice(self.episode_starts)
        idx = np.arange(start_idx, start_idx + size) % self.max_size
        for k, v in self.data.items():
            out[k] = v[idx]
        return out

    #def add(self, data: Dict[str, np.ndarray]) -> None:
    def add(self, **data) -> None:
        for k in self.data:
            assert k in data

        for k, v in data.items():
            assert k in self.data
            if v.ndim > 1 or self.data[k].ndim > 1:
                assert v.shape[1:] == self.data[k].shape[1:]

        batch_sizes = [d.shape[0] for d in data.values()]
        assert all(batch_sizes[0] == s for s in batch_sizes)
        batch_size = batch_sizes[0]

        idx = np.arange(self.ptr, self.ptr + batch_size) % self.max_size
        # overflow = self.ptr + batch_size >= self.max_size
        # Delete episode starts that we overwrite
        # TODO: We need to zero the data after the episode starts?
        # Or what happens? We will replay a partial episode
        # At most 1 partial episode will exist, can we ignore it?
        self.episode_starts = [
            e for e in self.episode_starts 
            if e < self.ptr or e > (self.ptr + batch_size) % self.max_size
        ]
        # New episode starts
        new_starts = (
            self.ptr + data[self.start_key].nonzero()[0]
        ) % self.max_size 
        self.episode_starts += new_starts.tolist()

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

        for k, v in data.items():
            self.data[k][idx] = np.array(v, copy=False)




if __name__ == '__main__':
    b = ReplayBuffer({
        "a": {"shape": (2,3), "dtype": np.int32},
        "b": {"shape": (3,4), "dtype": np.float32}
    }, 10)
    data = {
        "a": np.ones((5,2,3)),
        "b": np.ones((5,3,4))
    }
    b.extend(data)

    b.sample(2)

    b = TapeBuffer({
        "a": {"shape": (2,3), "dtype": np.int32},
        "b": {"shape": (3,4), "dtype": np.float32},
        "start": {"shape":(), "dtype": bool}
    }, 10, "start")
    start0 = np.array(
        [True, False, False, True, False]
    )
    data = {
        "a": np.arange(5*2*3).reshape((5,2,3)),
        "b": np.arange(5*3*4).reshape((5,3,4)),
        "start": start0
    }
    b.extend(data)

    start1 = np.array(
        [True, False]
    )
    data2 = {
        "a": data['a'].max() + 1 + np.arange(2*2*3).reshape((2,2,3)),
        "b": data['b'].max() + 1 + np.arange(2*3*4).reshape((2,3,4)),
        "start": start1,
    }
    b.extend(data2)
    b.sample(2)