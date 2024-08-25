import numpy as np
import jax.numpy as jnp

IMAGE_NORMALISATION = 255.0

def normalise_images(images: np.ndarray) -> jnp.ndarray:
    return jnp.array(images, dtype=jnp.float32) / IMAGE_NORMALISATION

def reshape_images(images: jnp.ndarray) -> jnp.ndarray:
    return images.reshape(-1, 28, 28, 1)

def count_parameters(variables):
    def count_params_recursive(params):
        total = 0
        if isinstance(params, dict):
            for value in params.values():
                total += count_params_recursive(value)
        else:
            total += jnp.prod(jnp.array(params.shape))
        return total

    return count_params_recursive(variables['params'])