from pathlib import Path
from typing import Tuple

import fire
import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey
from PIL import Image

from forward_process import calculate_alphas, get_noise_schedule
from model import UNet, initialize_model
from utils import load_state, ParamType
from train import NUM_TIMESTEPS

IMAGE_SHAPE = (1, 28, 28, 1)


def load_model(checkpoint_path: Path) -> Tuple[UNet, ParamType, ParamType]:
    state = load_state(checkpoint_path)
    model, _, _ = initialize_model(PRNGKey(0))
    parameters, batch_stats = state["params"], state["batch_stats"]
    return model, parameters, batch_stats


def calculate_mean(
    z: jnp.ndarray, beta: jnp.ndarray, alpha: jnp.ndarray, g: jnp.ndarray
) -> jnp.ndarray:
    prefactor = 1 / (1 - beta) ** 0.5
    weighted_prediction = (beta / (1 - alpha) ** 0.5) * g
    return prefactor * (z - weighted_prediction)


def run_ddpm(
    model: UNet,
    params: ParamType,
    batch_stats: ParamType,
    z_T: jnp.ndarray,
    alphas: jnp.ndarray,
    noise_schedule: jnp.ndarray,
    num_timesteps: int,
    prng_key: jax.Array,
) -> jnp.ndarray:
    z = z_T
    timestep_array = jnp.array([num_timesteps])
    for t in range(num_timesteps - 1, -1, -1):
        print(f"Running timestep {t}")
        alpha = alphas[t]
        beta = noise_schedule[t]
        g = model.apply(
            {"params": params, "batch_stats": batch_stats},
            z,
            timestep_array,
            train=False,
        )
        mu = calculate_mean(z, beta, alpha, g)
        epsilon = jax.random.normal(prng_key, IMAGE_SHAPE)
        z = mu + beta**0.5 * epsilon
    return z


def ddpm(
    model: UNet, params: ParamType, batch_stats: ParamType, num_timesteps: int
) -> jnp.ndarray:
    noise_schedule = get_noise_schedule(num_timesteps)
    alphas = calculate_alphas(num_timesteps)
    prng_key = jax.random.PRNGKey(0)
    z_T = jax.random.normal(prng_key, IMAGE_SHAPE)
    image = run_ddpm(
        model,
        params,
        batch_stats,
        z_T,
        alphas,
        noise_schedule,
        num_timesteps,
        prng_key,
    )
    return image


def get_image(checkpoint_path: Path) -> jnp.ndarray:
    model, params, batch_stats = load_model(checkpoint_path)
    return ddpm(model, params, batch_stats, NUM_TIMESTEPS)


def save_image_as_jpeg(image_array: jnp.ndarray, file_path: str) -> None:
    image = image_array.squeeze()
    image_np = np.array(image)
    image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
    image_pil.save(file_path, format="JPEG")


def main(checkpoint: str) -> None:
    checkpoint_path = Path(checkpoint)
    image = get_image(checkpoint_path)
    save_image_as_jpeg(image, "generated_image.jpg")


if __name__ == "__main__":
    fire.Fire(main)
