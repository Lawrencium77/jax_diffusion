from pathlib import Path
from typing import Any, Tuple

import fire
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from flax.core import FrozenDict
from jax.random import PRNGKey
from tqdm import tqdm

from forward_process import calculate_alphas, get_noise_schedule
from model import UNet, initialize_model
from utils import load_parameters, SPATIAL_DIM, NUM_CHANNELS
from train import NUM_TIMESTEPS


def load_model(checkpoint_path: Path) -> Tuple[UNet, Any]:
    model, _ = initialize_model()
    params = load_parameters(checkpoint_path)
    return model, params


def calculate_mean(
    z: jnp.ndarray, beta: jnp.ndarray, alpha: jnp.ndarray, g: jnp.ndarray
) -> jnp.ndarray:
    prefactor = 1 / (1 - beta) ** 0.5
    weighted_prediction = (beta / (1 - alpha) ** 0.5) * g
    return prefactor * (z - weighted_prediction)


def run_ddpm(
    model: UNet,
    params: FrozenDict[str, Any],
    z: jnp.ndarray,
    alphas: jnp.ndarray,
    noise_schedule: jnp.ndarray,
    output_shape: Tuple[int, int, int, int],
    key: jax.Array,
) -> jnp.ndarray:
    num_timesteps = len(alphas)
    for t in tqdm(reversed(range(num_timesteps))):
        alpha = alphas[t]
        beta = noise_schedule[t]
        g = model.apply({"params": params}, z, jnp.array([t]))
        if not isinstance(g, jnp.ndarray):
            raise ValueError("Model output is not a jnp.ndarray")

        key, subkey = jax.random.split(key)
        epsilon = (
            jnp.zeros(output_shape)
            if t == 0
            else jax.random.normal(subkey, output_shape)
        )

        z = calculate_mean(z, beta, alpha, g) + beta**0.5 * epsilon
    return z


def ddpm(
    model: UNet,
    params: FrozenDict[str, Any],
    num_timesteps: int,
    num_images: int,
    key: jnp.ndarray,
) -> jnp.ndarray:
    noise_schedule = get_noise_schedule(num_timesteps)
    alphas = calculate_alphas(num_timesteps)
    output_shape = (num_images, SPATIAL_DIM, SPATIAL_DIM, NUM_CHANNELS)
    z_T = jax.random.normal(key, output_shape)
    image = run_ddpm(
        model,
        params,
        z_T,
        alphas,
        noise_schedule,
        output_shape,
        key,
    )
    return image


def get_image(
    checkpoint_path: Path,
    num_images: int,
    key: jnp.ndarray,
) -> jnp.ndarray:
    model, params = load_model(checkpoint_path)
    return ddpm(model, params, NUM_TIMESTEPS, num_images, key)


def save_image_as_jpeg(
    image_array: jnp.ndarray,
    file_path: Path,
    num_images: int,
) -> None:
    images = np.array(image_array)
    for i in range(num_images):
        image = images[i].squeeze()
        output_path = file_path.with_name(f"{file_path.stem}_{i}{file_path.suffix}")
        plt.imsave(output_path, image, cmap="gray")


def main(
    checkpoint: str,
    output_path: str = "image.jpg",
    num_images: int = 1,
    key: jnp.ndarray = PRNGKey(0),
) -> None:
    image = get_image(Path(checkpoint), num_images, key)
    save_image_as_jpeg(image, Path(output_path), num_images)


if __name__ == "__main__":
    fire.Fire(main)
