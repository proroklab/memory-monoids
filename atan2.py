# import jax
# import jax.numpy as jnp

# @jax.custom_vjp
# def mul(y, x):
#     return y * x

# def mul_fwd(y, x):
#     return mul(y, x), None
    
# def mul_bwd(res, g):
#     return (g, g)
#     #breakpoint()
#     #return jax.grad(mul)(x, y)

# mul.defvjp(mul_fwd, mul_bwd)
# y = jnp.array(1.0)
# x = jnp.array(2.0)
# f = mul(y, x)
# g = jax.grad(mul)(y, x)

import jax
import jax.numpy as jnp

@jax.custom_vjp
def clamped_arctan2(y, x):
    return jnp.arctan2(y, x)

def clamped_arctan2_f(y, x):
    return clamped_arctan2(y, x), (y, x)

def clamped_arctan2_b(res, g):
    y, x = res
    # Compute the gradients with respect to y and x
    dy = jax.grad(jnp.arctan2, argnums=0)(y, x)
    dx = jax.grad(jnp.arctan2, argnums=1)(y, x)

    # Clip the gradients
    dy_clipped = jnp.clip(dy, a_min=-10.0, a_max=10.0)
    dx_clipped = jnp.clip(dx, a_min=-10.0, a_max=10.0)

    # Return the product of the clipped gradients and the upstream gradient
    return g * dy_clipped, g * dx_clipped

clamped_arctan2.defvjp(clamped_arctan2_f, clamped_arctan2_b)


if __name__ == '__main__':
    standard_fn = lambda y, x: jnp.arctan2(2 * y, x**2) 
    clamped_fn = lambda y, x: clamped_arctan2(2 * y, x**2) 

    # Test it does the right thing when the grad is small
    y = jnp.array(2.0)
    x = jnp.array(1.0)
    standard = standard_fn(y, x)
    clamped = clamped_fn(y, x)
    standard_g = jax.grad(standard_fn, argnums=0)(y, x).item(), jax.grad(standard_fn, argnums=1)(y, x).item()
    clamped_g = jax.grad(clamped_fn, argnums=0)(y, x).item(), jax.grad(clamped_fn, argnums=1)(y, x).item()

    print(standard, clamped)
    print(standard_g, clamped_g)

    # Test it does the right thing when the grad is big
    y = jnp.array(2e-15)
    x = jnp.array(1e-10)
    standard = standard_fn(y, x)
    clamped = clamped_fn(y, x)
    standard_g = jax.grad(standard_fn, argnums=0)(y, x).item(), jax.grad(standard_fn, argnums=1)(y, x).item()
    clamped_g = jax.grad(clamped_fn, argnums=0)(y, x).item(), jax.grad(clamped_fn, argnums=1)(y, x).item()

    print(standard, clamped)
    print(standard_g, clamped_g)