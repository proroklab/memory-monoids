from typing import List, Tuple, Union
import equinox as eqx
from jax import Array, lax
import jax.numpy as jnp
from jax.random import PRNGKey


class MemoryModule(eqx.Module):
    """Base class for a memory module. 
    
    This will sit within a Q network to compute recurrent states. 
    It consists of four principle functions:

    1. associative_update: H x H -> H    
        An associative update for the recurrent state 

    2. wrapped_associative_update: (H, b) x (H, b) -> (H, b)
        A reset-wrapped version of #1, which can reset the recurrent state using the start/begin flag

    3. scan: H x O -> H
        An inner function that given the previous recurrent state and observations, computes the next recurrent state.
        This should call wrapped_associative_update internally.

    4. __call__: O x H -> S x H
        An outer function that takes an observation and previous recurrent state and computes the markov state.
        This function should call aggregate internally.
    """

    def __call__(self, x: Array, state: Array, start: Array, next_done: Array, key: PRNGKey=None) -> Union[Array, List[Array]]:
        """Forward pass of the model, which should call aggregate.

        This is known as M in the paper:
        O x H -> S x H."""
        return NotImplementedError()

    def initial_state(self, shape: Tuple[int, ...] = tuple()) -> List[Array]:
        """Return the recurrent state for the beginning of a sequence.

        NOTE: Due to how lax.associative_scan is implemented, all states MUST have
        the same number of dimensions. For example, rather than
        returning two arrays ([JxK], [L]), you must return ([JxK], [1xL])."""
        return NotImplementedError()

    def associative_update(
        self,
        carry: Tuple[Array, Array, Array],
        incoming: Tuple[Array, Array, Array],
    ) -> Tuple[Array, ...]:
        """Update the recurrent state, to be invoked in MemoryModel.aggregate using lax.associative_scan.

        This is the bullet (monoid binary operator) in the paper.
        H o H -> H"""
        raise NotImplementedError()

    # Currently, next_done points to the final obs in an episode
    # and start points to the initial obs in an episode
    # obs:       [0, 1, 1, 0, 1] (here the zero is the initial obs of an episode)
    # start:     [1, 0, 0, 1, 0]
    # prev_start:[0, 1, 0, 0, 1]
    # done:      [0, 0, 0, 1, 0]
    # next_done: [0, 0, 1, 0, 0]
    # During inference, "next_done" will never be received because we only
    # ever see one step/episode at a time.
    # This is probably what we want (to only use next_done during training)
    def wrapped_associative_update(self, carry: Array, incoming: Array) -> Tuple[Array, ...]:
        """The reset-wrapped form of the associative update. 

        You might need to override this
        if you use variables in associative_update that are not from initial_state. 
        This is equivalent to the h function in the paper:
        b x H -> b x H
        """
        prev_start, *carry = carry
        start, *incoming = incoming
        # Reset all elements in the carry if we are starting a new episode
        initial_states = self.initial_state((start.shape[0],))
        assert len(initial_states) == len(carry)
        carry = tuple([state * jnp.logical_not(start) + start * i_state for state, i_state in zip(carry, initial_states)])
        out = self.associative_update(carry, incoming)
        start_out = jnp.logical_or(start, prev_start)
        return (start_out, *out)

    def scan(
        self,
        x: Array,
        state: List[Array],
        start: Array,
    ) -> Array:
        """Given an input and recurrent state, this will update the recurrent state. 

        This is equivalent to the inner-function g in the paper:
        H x O -> H
        """
        # x: [T, memory_size]
        # memory: [1, memory_size, context_size]
        T = x.shape[0]
        # Match state_dim
        dims = max([len(s.shape[1:]) for s in state])
        start = start.reshape(T, *[1 for _ in range(dims)])

        # Now insert previous recurrent state
        x = jnp.concatenate([state, x], axis=0)
        start = jnp.concatenate([jnp.zeros_like(start[:1]), start], axis=0)

        # This is not executed during inference -- method will just return x if size is 1
        _, new_state = lax.associative_scan(
            self.wrapped_associative_update,
            (start, x),
            axis=0,
        )
        return new_state[1:]