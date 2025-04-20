import numpy as np
import jax
import jax.numpy as jnp

@jax.jit
def compute(x):
    return jnp.sin(x) + jnp.log(x)

x_np = np.ones((1000, 1000))     # NumPy array (CPU)
x_jax = jnp.array(x_np)          # Convert to JAX array (on GPU if available)

result = compute(x_jax)
print(result)