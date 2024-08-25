import numpy as np
import jax.numpy as jnp

IMAGE_NORMALISATION = 255.0

def normalise_images(images: np.ndarray) -> jnp.ndarray:
    return jnp.array(images, dtype=jnp.float32) / IMAGE_NORMALISATION

def reshape_images(images: jnp.ndarray) -> jnp.ndarray:
    return images.reshape(-1, 28, 28, 1)