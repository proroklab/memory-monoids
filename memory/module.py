from typing import List, Tuple, Union
import equinox as eqx
from jax import Array
import jax.numpy as jnp
from jax.random import PRNGKey


class MemoryModule(eqx.Module):
    """Base class for a memory module. This will sit within a Q network
    to compute recurrent states."""

    def __call__(self, x: Array, state: Array, start: Array, next_done: Array, key: PRNGKey=None) -> Union[Array, List[Array]]:
        """Forward pass of the model"""
        return NotImplementedError()

    def initial_state(self, shape: Tuple[int, ...] = tuple()) -> List[Array]:
        """Return the recurrent state for the beginning of a sequence"""
        return NotImplementedError()

# Each file should also contain associative_update
# and reset wrapped associative update.
# We can't use standard instance methods utilizing self
# it makes the jax.compiler slow and sad :(

def associative_update(
    carry: Tuple[Array, Array, Array],
    incoming: Tuple[Array, Array, Array],
) -> Tuple[Array, ...]:
    """Update the recurrent state, to be invoked in MemoryModel.__call__ using lax.associative_scan"""
    raise NotImplementedError()

def initial_state(shape: Tuple[int, ...] = tuple()) -> List[Array]:
    """Return the recurrent state for the initial timestep of a sequence"""
    return NotImplementedError()

def wrap_associative_update_in_reset(associative_update_fn: callable, initial_state_fn: callable) -> Tuple[Array, ... ]:
    """Wrap the associative update in an automatic reset. You might have to override this if you
    are doing something less-than-standard."""

    def associative_update(carry: Array, incoming: Array) -> Tuple[Array, ...]:
        prev_start, *carry = carry
        start, *incoming = incoming
        # Reset all elements in the carry if we are starting a new episode
        initial_states = initial_state_fn((start.shape[0],))
        assert len(initial_states) == len(carry)
        carry = tuple([state * jnp.logical_not(start) + start * i_state for state, i_state in zip(carry, initial_states)])
        out = associative_update_fn(carry, incoming)
        start_out = jnp.logical_or(start, prev_start)
        return tuple(start_out, *out)

    return associative_update


