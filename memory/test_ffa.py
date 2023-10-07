import ffa
import jax
import jax.numpy as jnp

def one_episode_loop(params, x, state, next_done):
    xs = []
    i = 0
    one = jnp.array([1])
    for i in range(len(next_done)):
        state = state * ffa.gamma(params, one) + jnp.expand_dims(x[i:i+1], -1)
        xs.append(state)
        if next_done[i]:
            break
    return jnp.concatenate(xs)

def recurrent_loop(params, x, state, start, next_done):
    xs = []
    i = 0
    one = jnp.array([1])
    for i in range(len(next_done)):
        state = state * ffa.gamma(params, one) * jnp.logical_not(start[i]) + jnp.expand_dims(x[i:i+1], -1)
        xs.append(state)
    return jnp.concatenate(xs)

def check_apply(params, new_state, x, state, start, done):
    desired = recurrent_loop(params, x, state, start, done)
    # idx = 0
    # desired = []
    # while True:
    #     ep = one_episode_loop(params, x[idx:], jnp.zeros_like(state), done[idx:])
    #     idx += ep.shape[0]
    #     desired.append(ep)
    #     if idx >= x.shape[0]:
    #         break
    # desired = jnp.concatenate(desired)

    # ep1 = one_episode_loop(params, x, jnp.zeros_like(state), done)
    # idx = ep1.shape[0]
    # ep2 = one_episode_loop(params, x[idx:], jnp.zeros_like(state), done[idx:])
    # idx = idx + ep2.shape[0]
    # ep3 = one_episode_loop(params, x[idx:], jnp.zeros_like(state), done[idx:])
    # desired = jax.lax.stop_gradient(jnp.concatenate([ep1, ep2, ep3]))
    # actual = jax.lax.stop_gradient(new_state[:desired.shape[0]])
    print((desired - new_state).squeeze(-1).squeeze(-1))
    print((desired - new_state).squeeze(-1).squeeze(-1).real.max())
    breakpoint()

if __name__ == '__main__':
    size = 1000
    context_size = 1
    memory_size = 1
    params = (jnp.array([-0.1]), jnp.array([jnp.pi / 4]))
    with jax.disable_jit():
        key = jax.random.PRNGKey(0)
        #params = ffa.init(memory_size=memory_size, context_size=context_size)
        x = jnp.ones((size, memory_size), dtype=jnp.float32)
        s = ffa.initial_state(params)
        #s = (jnp.zeros((memory_size, context_size), dtype=jnp.complex64))
        start = jax.random.uniform(key, (size,)) > 0.95
        next_done = jnp.concatenate([start[1:], jnp.array([False])])
        #start = jnp.zeros((size,), dtype=bool)
        result = ffa.apply(params, x, s, start, next_done)
        check_apply(params, result, x, s, start, next_done)