from pathlib import Path
from typing import Optional, Tuple

import fire
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from jax.random import PRNGKey
from tqdm import tqdm

from forward_process import calculate_alphas, get_noise_schedule
from model import UNet, initialize_model
from utils import load_state, ParamType, SPATIAL_DIM, NUM_CHANNELS
from train import NUM_TIMESTEPS


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
    output_shape: Tuple[int, int, int, int],
    key: jax.Array,
) -> jnp.ndarray:
    z = z_T
    for t in tqdm(range(num_timesteps - 1, -1, -1)):
        alpha = alphas[t]
        beta = noise_schedule[t]
        timestep_array = jnp.array([t])
        g = model.apply(
            {"params": params, "batch_stats": batch_stats},  # type: ignore
            z,
            timestep_array,
            train=False,
        )
        key, subkey = jax.random.split(key)
        if t > 0:
            epsilon = jax.random.normal(subkey, output_shape)
            z = calculate_mean(z, beta, alpha, g) + beta**0.5 * epsilon  # type: ignore
        else:
            z = calculate_mean(z, beta, alpha, g)  # type: ignore
    return z


def ddpm(
    model: UNet,
    params: ParamType,
    batch_stats: ParamType,
    num_timesteps: int,
    num_images: int,
    key: Optional[jax.Array],
) -> jnp.ndarray:
    noise_schedule = get_noise_schedule(num_timesteps)
    alphas = calculate_alphas(num_timesteps)
    if key is None:
        key = jax.random.PRNGKey(0)
    output_shape = (num_images, SPATIAL_DIM, SPATIAL_DIM, NUM_CHANNELS)
    z_T = jax.random.normal(key, output_shape)
    image = run_ddpm(
        model,
        params,
        batch_stats,
        z_T,
        alphas,
        noise_schedule,
        num_timesteps,
        output_shape,
        key,
    )
    return image


def get_image(
    checkpoint_path: Path,
    num_images: int,
    key: Optional[jax.Array] = None,
) -> jnp.ndarray:
    model, params, batch_stats = load_model(checkpoint_path)
    return ddpm(model, params, batch_stats, NUM_TIMESTEPS, num_images, key)


def save_image_as_jpeg(
    image_array: jnp.ndarray,
    file_path: Path,
    num_images: int,
) -> None:
    for i in range(num_images):
        image = image_array[i]
        image = np.array(image)
        image = image.squeeze()
        output_path = file_path.with_name(file_path.stem + f"_{i}" + file_path.suffix)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.savefig(output_path)


def main(
    checkpoint: str,
    output_path: str = "image.jpg",
    num_images: int = 1,
) -> None:
    image = get_image(Path(checkpoint), num_images)
    save_image_as_jpeg(image, Path(output_path), num_images)


if __name__ == "__main__":
    fire.Fire(main)
