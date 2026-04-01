import numpy as np

class Backend:

    _instance = None

    def __new__(cls):

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            
            try:
                import jax
                import jax.numpy as jnp
                cls._instance.xp = jnp
                cls._instance.type = 'jax'
                cls._instance.jit = jax.jit
                #from jax import config
                #config.update("jax_enable_x64", True)
            except ImportError:
                cls._instance.xp = np
                cls._instance.type = 'numpy'
                cls._instance.jit = lambda f: f  # does nothing
    
        return cls._instance

backend = Backend()