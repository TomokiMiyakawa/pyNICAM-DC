import jax
import jax.numpy as jnp

x = jnp.ones((3, 3))
print("x is on:", x.addressable_data(0).device)