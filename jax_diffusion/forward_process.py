"""
Various functions for the forward and reverse diffusion processes.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax.random import PRNGKey

MIN_NOISE = 1e-4
MAX_NOISE = 0.02


def get_noise_schedule(num_timesteps: int) -> jnp.ndarray:
    """
    Linear noise schedule, as used in https://arxiv.org/pdf/2006.11239.
    """
    return jnp.linspace(MIN_NOISE, MAX_NOISE, num_timesteps)


def calculate_alphas(num_timesteps: int) -> jnp.ndarray:
    noise_schedule = get_noise_schedule(num_timesteps)
    alphas = jnp.cumprod(1 - noise_schedule)
    return alphas


def sample_latents(
    images: jnp.ndarray,
    num_timesteps: int,
    alphas: jnp.ndarray,
    key_t: PRNGKey,
    key_n: PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Sample from forward process.
    """
    batch_size = images.shape[0]
    timesteps = jax.random.randint(
        key_t, (batch_size,), 0, num_timesteps
    )
    sampled_alphas = alphas[timesteps].reshape(batch_size, 1, 1, 1)
    noise = jax.random.normal(key_n, images.shape)
    latents = (
        jnp.sqrt(sampled_alphas) * images
        + jnp.sqrt(1 - sampled_alphas) * noise
    )
    return latents, noise, timesteps
