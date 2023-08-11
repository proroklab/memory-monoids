from torch import nn
import torch
from tensordict.nn import TensorDictModuleBase
from typing import Optional, Tuple, List
from torchrl.envs import TensorDictPrimer
from torchrl.data import UnboundedContinuousTensorSpec


def prepend_zero(x, dim=0):
    shape = x.shape[:dim] + (1,) + x.shape[dim + 1 :]
    zero = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([zero, x], dim=dim)

def append_zero(x, dim=0):
    shape = x.shape[:dim] + (1,) + x.shape[dim + 1 :]
    zero = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return torch.cat([x, zero], dim=dim)

def kernel_space_reset(state, done, time_dim=0):
    # Do we have a done at the next timestep?
    terminal_state_mask = append_zero(done[1:])
    # Each transition assigned index 0 to k
    sequence_idx = done.cumsum(time_dim)
    # If the initial done is true, then we need to be careful as the sequence
    # indices will start at 1 instead of 0
    sequence_idx = sequence_idx - sequence_idx.min()
    # Get the final state of each sequence
    # If "done", it means that the current state corresponds to the next sequence
    # so the terminal states should be the states before done
    terminal_states = state[terminal_state_mask]
    # We don't want to reset the first sequence, so prepend zeros
    # for the previous (non-visible) terminal state
    terminal_states = prepend_zero(terminal_states)
    #resetter = terminal_states[sequence_idx]
    resetter = terminal_states.index_select(time_dim, sequence_idx)
    reset_state = state - resetter
    return reset_state


class LinearAttentionModule(TensorDictModuleBase):
    def __init__(self, input_size, hidden_size, in_key="embed", recurrent_keys=["recurrent_state_s", "recurrent_state_z"], out_key="action_value", done_key=("collector", "mask")):
        super().__init__()
        self.in_keys = [in_key, *recurrent_keys, done_key]
        self.recurrent_keys = recurrent_keys
        self.done_key = done_key
        out_rkeys = [("next", key) for key in recurrent_keys]
        self.out_keys = [out_key, *out_rkeys]
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.module = LinearAttentionBlock(input_size, hidden_size)

    def forward(self, td):
        x = td.get(self.in_keys[0])
        s_state = td.get(self.recurrent_keys[0])
        z_state = td.get(self.recurrent_keys[1])

        num_squeezes = 0
        while x.dim() < 3:
            x = x.unsqueeze(0)
            s_state = s_state.unsqueeze(0)
            z_state = z_state.unsqueeze(0)
            num_squeezes += 1

        y, state = self.module(x, [s_state, z_state])
        s_state, z_state = state

        for _ in range(num_squeezes):
            y = y.squeeze(0)
            s_state = s_state.squeeze(0)
            z_state = z_state.squeeze(0)

        td[self.out_keys[0]] = y
        td[self.out_keys[1]] = s_state
        td[self.out_keys[2]] = z_state
        return td

    def temporal_mode(self, value):
        return self

    def make_tensordict_primer(self):
        return TensorDictPrimer(
            {
                self.recurrent_keys[0]: UnboundedContinuousTensorSpec(
                    shape=(self.hidden_size, self.hidden_size),
                ),
                self.recurrent_keys[1]: UnboundedContinuousTensorSpec(

                    shape=(1, self.hidden_size),
                )
            }
        )

class LinearAttentionBlock(nn.Module):
    """
    The building block from the Linear Transformers are Secretly RNNs Paper. This is
    a form of linear transformer.

    Inputs:
        input_size: Size of input feature dim
        hidden_size: Size of key/query/value space
        S_aggregator: Which type of aggregation to use for the numerator (S term)
        Z_aggregator: Which type of aggregation to use for the denominator (Z term)
        feed_forward: Whether to apply a perceptron to the output
        residual: Whether to apply a residual connection from input to output
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        feed_forward=True,
        residual=True,
        normalize=False,
    ):
        super().__init__()
        self.key = nn.Linear(input_size, hidden_size, bias=False)
        self.query = nn.Linear(input_size, hidden_size, bias=False)
        self.value = nn.Linear(input_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(input_size)
        self.phi = nn.ELU()
        self.feed_forward = feed_forward
        self.residual = residual
        self.normalize = normalize

        if self.feed_forward:
            self.ff = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True)
            )
        if self.residual:
            self.shortcut = nn.Linear(input_size, hidden_size)

    def forward(
        self, x: torch.Tensor, state: List[torch.Tensor], done: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Input:
            x: [B, T, F]
            state: Tuple[
                [B, 1, D, D],
                [B, 1, D]
            ]
        Output:
            y: [B, T, D]
            state: Tuple[
                [B, 1, D, D],
                [B, 1, D]
            ]
        """

        #done = done or torch.zeros(x.shape[:2], device=x.device, dtype=torch.bool)
        x = self.norm(x)
        K = self.phi(self.key(x))
        Q = self.phi(self.query(x))
        if self.normalize:
            K = K / K.norm(dim=-1, keepdim=True)
            Q = Q / Q.norm(dim=-1, keepdim=True)
        V = self.value(x)
        S, Z = state
        B, T, F = K.shape

        # S = sum(K V^T)
        S = (torch.einsum("bti, btj -> btij", K, V).cumsum(dim=-3) + S)
        Z = (K.reshape(B, T, 1, F) + Z.cumsum(dim=-3))
        if done:
            # TODO S and Z are off by one because of the added S and Z at the beginning
            S = kernel_space_reset(S, done)
            Z = kernel_space_reset(Z, done)

        # S = self.S_aggregator(
        #     torch.einsum("bti, btj -> btij", K, V).reshape(B, T, F * F),
        #     S.reshape(B, 1, F * F),
        # ).reshape(B, T, F, F)
        # Z = sum(K)
        # Z = self.Z_aggregator(K, Z.reshape(B, 1, F))
        # numerator = Q^T S
        numerator = torch.einsum("bti, btil -> btl", Q, S)
        # denominator = Q^T Z
        denominator = torch.einsum("bti, btl -> bt", Q, Z.reshape(B, T, F)).reshape(B, T, 1) 
        denominator = torch.sign(denominator) * torch.clamp(torch.abs(denominator), min=1e-8)
        # output = (Q^T S) / (Q^T Z)
        output = numerator / denominator

        if self.feed_forward:
            output = self.ff(output)

        if self.residual:
            output = output + self.shortcut(x)

        state = [S, Z]

        return output, state