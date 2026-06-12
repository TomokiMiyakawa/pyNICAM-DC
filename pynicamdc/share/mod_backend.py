import numpy as np

class Backend:
    
    _instance = None

    def __init__(self):
        self.configured = False
        self.type = None
        self.xp = None
        self.dtype = None

    def configure(self, backend_name, precision):
        self.np = np  # always have numpy available
        self.ndtype = np.float32 if precision == "float32" else np.float64

        if backend_name == "numpy":
            self.type = "numpy"
            self.xp = np
            self.dtype = self.ndtype
            self.to_numpy = lambda x: np.asarray(x)

        elif backend_name == "jax":
            import jax
            jax.config.update("jax_enable_x64", precision == "float64")
            import jax.numpy as jnp

            self.type = "jax"
            self.jax = jax
            self.xp = jnp
            self.dtype = jnp.float32 if precision == "float32" else jnp.float64
            self.to_numpy = lambda x: np.asarray(x)

backend = Backend()
