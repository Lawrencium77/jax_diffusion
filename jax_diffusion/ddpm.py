from pathlib import Path
from typing import Optional, Tuple

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
    key: jax.Array,
) -> jnp.ndarray:
    z = z_T
    for t in range(num_timesteps - 1, -1, -1):
        print(f"Running timestep {t}")
        alpha = alphas[t]
        beta = noise_schedule[t]
        timestep_array = jnp.array([t])
        g = model.apply(
            {"params": params, "batch_stats": batch_stats},
            z,
            timestep_array,
            train=False,
        )
        key, subkey = jax.random.split(key)
        if t > 0:
            epsilon = jax.random.normal(subkey, IMAGE_SHAPE)
            z = calculate_mean(z, beta, alpha, g) + beta**0.5 * epsilon
        else:
            z = calculate_mean(z, beta, alpha, g)
    return z


def ddpm(
    model: UNet,
    params: ParamType,
    batch_stats: ParamType,
    num_timesteps: int,
    key: Optional[jax.Array],
) -> jnp.ndarray:
    noise_schedule = get_noise_schedule(num_timesteps)
    alphas = calculate_alphas(num_timesteps)
    if key is None:
        key = jax.random.PRNGKey(0)
    z_T = jax.random.normal(key, IMAGE_SHAPE)
    image = run_ddpm(
        model,
        params,
        batch_stats,
        z_T,
        alphas,
        noise_schedule,
        num_timesteps,
        key,
    )
    return image


def get_image(checkpoint_path: Path, key: Optional[jax.Array] = None) -> jnp.ndarray:
    model, params, batch_stats = load_model(checkpoint_path)
    return ddpm(model, params, batch_stats, NUM_TIMESTEPS, key)


def save_image_as_jpeg(image_array: jnp.ndarray, file_path: str) -> None:
    image = image_array.squeeze()
    image_np = np.array(image)
    image_normalised = (((image_np + 1.0) / 2.0) * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_normalised)
    image_pil.save(file_path, format="JPEG")


def main(checkpoint: str, output_path: str = "generated_image.jpg") -> None:
    checkpoint_path = Path(checkpoint)
    image = get_image(checkpoint_path)
    save_image_as_jpeg(image, output_path)


if __name__ == "__main__":
    fire.Fire(main)
