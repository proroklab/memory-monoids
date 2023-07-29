from torch import nn
from copy import deepcopy
from tensordict.nn import make_functional, TensorDictModuleBase


class SoftUpdateModule(TensorDictModuleBase):
    def __init__(self, module: TensorDictModuleBase, tau: float = 0.05):
        super().__init__()
        self.module = module
        self.params = make_functional(module, keep_params=True)
        breakpoint()
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys
        self.target_params = self.params.detach().clone()
        self.tau = tau

    def forward(self, td):
        return self.module(td, params=self.target_params)

    def step(self):
        for param, target_param in zip(self.params, self.target_params):
            target_param.set_(target_param * (1.0 - self.tau) + param * self.tau)


