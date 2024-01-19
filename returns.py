import jax
from jax import numpy as jnp
import numpy as np
import time
import pandas as pd


episode_lengths = [10, 100, 1000]
batch_sizes = [10, 100, 1000]
gamma = 0.99

@jax.jit
def serial_discounted_return(rewards):
    gammas = gamma ** jnp.arange(rewards.size)
    return jnp.cumsum(rewards * gammas)


@jax.jit
def discounted_return_op(carry, inputs):
    del_a, r_a, flag_a = carry
    del_b, r_b, flag_b = inputs
    del_a = del_a * jnp.logical_not(flag_b) + flag_b * jnp.ones_like(del_a)
    r_a = r_a * jnp.logical_not(flag_b)
    flag_b = jnp.logical_or(flag_a, flag_b)
    out = (del_a * del_b, r_a + del_a * r_b, flag_b)
    return out


@jax.jit
def paralell_discounted_return(rewards, begin):
    d = jnp.ones(rewards.size) * gamma
    return jax.lax.associative_scan(discounted_return_op, (d, rewards, begin))



def run(silent=False):
    dfs = []
    for episode_length in episode_lengths:
        for batch_size in batch_sizes:
            lens = jax.random.randint(jax.random.PRNGKey(0), (batch_size,), 1, episode_length)
            rewards = jax.random.uniform(jax.random.PRNGKey(0), (jnp.sum(lens),))
            begin = jnp.zeros(rewards.size, dtype=bool)
            ends = jnp.cumsum(lens)
            starts = jnp.concatenate([jnp.array([0]), ends])[:-1]
            #starts, ends = jnp.cumsum(lens[:-1]), jnp.cumsum(lens[1:])
            begin = begin.at[starts].set(True)

            start = time.time()
            d_returns = []
            for s, e in zip(starts, ends):
                r = rewards[s:e]
                d_return = serial_discounted_return(r)
                d_returns.append(d_return)
            d_returns = jnp.concatenate(d_returns)
            d_returns.block_until_ready()
            total = (time.time() - start) * 1000
            if not silent:
                #print(f"Serial: {batch_size}, {episode_length}: {total}")
                dfs.append(pd.DataFrame(
                data={"Mode": "Serial", "Batch Size": batch_size, "Max Ep. Length": episode_length, "Time (ms)": total},
                index=[0]
                ))


            start = time.time()
            _, pd_returns, _ = paralell_discounted_return(rewards, begin)
            pd_returns.block_until_ready()
            total = (time.time() - start) * 1000
            if not silent:
                #print(f"Parallel: {batch_size}, {episode_length}: {total}")
                dfs.append(pd.DataFrame(
                data={"Mode": "Parallel", "Batch Size": batch_size, "Max Ep. Length": episode_length, "Time (ms)": total},
                index=[0]
                ))

            assert jnp.allclose(d_returns, pd_returns, atol=1e-5)
    return dfs

# Run once to compile
# Run twice to log
run(silent=True)
out_df = pd.DataFrame()
for i in range(5):
    dfs = run()
    print(dfs)
    out_df = pd.concat([out_df, *dfs], ignore_index=True)
out_df.to_csv('return_perf.csv')
breakpoint()


