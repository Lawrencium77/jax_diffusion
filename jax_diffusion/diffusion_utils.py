"""
Various functions for the forward and reverse diffusion processes.
"""

from typing import Tuple
import jax
import jax.numpy as jnp

def get_noise_schedule(num_timesteps: int) -> jnp.ndarray:
    noise_schedule = jnp.linspace(0, 1, num_timesteps)
    return noise_schedule

def calculate_alphas(num_timesteps: jnp.ndarray) -> jnp.ndarray:
    noise_schedule = get_noise_schedule(num_timesteps)
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
    sampled_alphas = alphas[timesteps][:, None] # Reshape for broadcasting
    noise = jax.random.normal(jax.random.PRNGKey(0), images.shape)
    latents = (sampled_alphas ** 0.5) * images + (1 - sampled_alphas) ** 0.5 * noise
    return latents, noise, timesteps

# if __name__ == "__main__":
#     num_timesteps = 1000
#     noise_schedule = get_noise_schedule(num_timesteps)
#     alphas = calculate_alphas(noise_schedule)
#     image = jnp.zeros((28, 28))
#     latent, noise, timestep = sample_latent(image, num_timesteps, alphas)
#     print(latent.shape, noise.shape, timestep)
#     print("Forward process complete.")