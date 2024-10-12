from pathlib import Path
from typing import Any, Dict, Union
import flax.serialization
import jax.numpy as jnp
import numpy as np

from flax.training.train_state import TrainState

SPATIAL_DIM = 32  # dataset.py resizes from 28 x 28
NUM_CHANNELS = 1


def normalise_images(images: np.ndarray) -> np.ndarray:
    return (images / 127.5) - 1.0


def reshape_images(images: np.ndarray) -> np.ndarray:
    return images.reshape(-1, SPATIAL_DIM, SPATIAL_DIM, NUM_CHANNELS)


def count_params(params) -> int:
    if isinstance(params, jnp.ndarray):
        return jnp.prod(jnp.array(params.shape)).item()
    return sum(count_params(value) for value in params.values())


def save_state(state: Union[TrainState, Dict[str, Any]], file_path: Path) -> None:
    state_bytes = flax.serialization.to_bytes(state)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(state_bytes)
    print(f"Saved state to {file_path}")


def load_state(file_path: Path) -> Dict[str, Any]:
    if not file_path.exists():
        raise FileNotFoundError(f"No parameter file found at {file_path}")

    state_bytes = file_path.read_bytes()
    state = flax.serialization.from_bytes(None, state_bytes)

    print(f"Loaded state from {file_path}")
    return state
