from pathlib import Path
from typing import Any, Dict, Union
import flax.serialization
import jax.numpy as jnp
import numpy as np

from flax.core import FrozenDict

IMAGE_NORMALISATION = 255.0

ParamType = Union[
    FrozenDict[str, Any],
    Dict[str, Any],
    jnp.ndarray,
]


def normalise_images(images: np.ndarray) -> jnp.ndarray:
    return jnp.array(images, dtype=jnp.float32) / IMAGE_NORMALISATION


def reshape_images(images: jnp.ndarray) -> jnp.ndarray:
    return images.reshape(-1, 28, 28, 1)


def count_params(params: ParamType) -> jnp.ndarray:
    total = jnp.array(0)
    if isinstance(params, jnp.ndarray):
        total += jnp.prod(jnp.array(params.shape))
    else:
        for value in params.values():
            total += count_params(value)
    return total


def save_model_parameters(parameters: ParamType, file_path: Path) -> None:
    param_bytes = flax.serialization.to_bytes(parameters)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(param_bytes)
    print(f"Saved model parameters to {file_path}")


def load_model_parameters(file_path: Path) -> ParamType:
    if not file_path.exists():
        raise FileNotFoundError(f"No parameter file found at {file_path}")

    param_bytes = file_path.read_bytes()
    parameters = flax.serialization.from_bytes(None, param_bytes)
    print(f"Loaded model parameters from {file_path}")
    return parameters
