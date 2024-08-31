from pathlib import Path
from typing import Dict, Union
import flax.serialization
import jax.numpy as jnp
import numpy as np

IMAGE_NORMALISATION = 255.0

NestedDict = Dict[str, Dict]
ParamDict = Dict[str, jnp.ndarray]
Params = Union[NestedDict, ParamDict, jnp.ndarray]


def normalise_images(images: np.ndarray) -> jnp.ndarray:
    return jnp.array(images, dtype=jnp.float32) / IMAGE_NORMALISATION


def reshape_images(images: jnp.ndarray) -> jnp.ndarray:
    return images.reshape(-1, 28, 28, 1)


def count_params(params: Params) -> int:
    total = 0
    if isinstance(params, dict):
        for value in params.values():
            total += count_params(value)
    else:
        total += jnp.prod(jnp.array(params.shape))
    return total


def save_model_parameters(parameters: NestedDict, file_path: Path) -> None:
    param_bytes = flax.serialization.to_bytes(parameters)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(param_bytes)
    print(f"Saved model parameters to {file_path}")


def load_model_parameters(file_path: Path) -> NestedDict:
    if not file_path.exists():
        raise FileNotFoundError(f"No parameter file found at {file_path}")

    param_bytes = file_path.read_bytes()
    parameters = flax.serialization.from_bytes(None, param_bytes)
    print(f"Loaded model parameters from {file_path}")
    return parameters
