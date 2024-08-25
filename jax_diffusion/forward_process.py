"""
Various functions for the forward and reverse diffusion processes.
"""

from typing import Tuple
from tqdm import tqdm

import jax
import jax.numpy as jnp

from jax import jit
from utils import normalise_images

@jit
def get_max_noise_batch(images: jnp.ndarray) -> float:
    images = normalise_images(images)
    images = images.reshape(images.shape[0], -1)
    distance_matrix = jnp.linalg.norm(images[:, None] - images[None, :], axis=-1)
    batch_max = jnp.max(distance_matrix)
    return batch_max

def get_max_noise(train_generator: jnp.ndarray) -> float:
    """
    Maximum noise is set to be comparable to the maximum pairwise distance between 
    all training data points.
    See https://arxiv.org/pdf/2006.09011.
    """
    print(f">>>>> Computing Max Noise Scale from Training Set <<<<<")
    max_noise = 0.0
    for images, _ in tqdm(train_generator):
        batch_max = get_max_noise_batch(images)
        max_noise = max(max_noise, batch_max)
    return max_noise


def get_noise_schedule(num_timesteps: int, max_noise: float) -> jnp.ndarray:
    """
    Use geometric progression for noise schedule, as suggested in 
    https://yang-song.net/blog/2021/score/
    """
    min_noise = 1e-4
    ratio = (max_noise / min_noise) ** (1.0 / (num_timesteps - 1))
    noise_schedule = min_noise * (ratio ** jnp.arange(num_timesteps))
    return noise_schedule


def calculate_alphas(train_generator, num_timesteps) -> jnp.ndarray:
    max_noise = get_max_noise(train_generator)
    noise_schedule = get_noise_schedule(num_timesteps, max_noise)
    alphas = jnp.cumprod(1 - noise_schedule)
    return alphas

def sample_latents(
    images: jnp.ndarray, 
    num_timesteps: jnp.ndarray, 
    alphas: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """
    Sample from forward process.
    """
    batch_size = images.shape[0]
    timesteps = jax.random.randint(jax.random.PRNGKey(0), (batch_size,), 0, num_timesteps)
    sampled_alphas = alphas[timesteps].reshape(batch_size, 1, 1, 1) 
    noise = jax.random.normal(jax.random.PRNGKey(0), images.shape)
    latents = (sampled_alphas ** 0.5) * images + (1 - sampled_alphas) ** 0.5 * noise
    return latents, noise, timesteps
