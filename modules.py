from typing import List, Union
from tensordict.nn import TensorDictModule, TensorDictModuleBase
import torch

class MaskModule(TensorDictModuleBase):
    def __init__(self, mask_key: Union[List[str], str]):
        super().__init__()
        self.mask_key = mask_key
        self.in_keys = [mask_key]
        self.out_keys = []

    def forward(self, td):
        mask = td.get(self.mask_key)
        old_td = td.clone()
        new_td = td.masked_select(mask).clone()
        #return new_td
        return new_td

class MinReduce(TensorDictModuleBase):
    def __init__(self, reduce_key: Union[List[str], str], reduce_index=0):
        super().__init__()
        self.reduce_key = reduce_key
        self.reduce_index = 0
        self.in_keys = [reduce_key]
        self.out_keys = []

    def forward(self, td):
        min_idx = td.get(self.reduce_key).argmin(self.reduce_index)
        breakpoint()
        return td[min_idx]