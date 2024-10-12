from functools import partial
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
import fire
from tqdm import tqdm

from dataset import NumpyLoader, get_dataset
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from forward_process import sample_latents, calculate_alphas
from jax import value_and_grad, jit
from jax.random import PRNGKey
from model import UNet, initialize_model
from typing import Any, Optional, Tuple
from utils import (
    count_params,
    load_parameters,
    normalise_images,
    reshape_images,
    save_state,
)

BATCH_SIZE = 128
NUM_TIMESTEPS = 1000


def get_optimiser(
    learning_rate: float = 0.001,
) -> optax.GradientTransformation:
    return optax.adam(learning_rate)


def get_loss(
    params: FrozenDict[str, Any],
    latents: jnp.ndarray,
    noise_values: jnp.ndarray,
    timesteps: jnp.ndarray,
    model: UNet,
) -> jnp.ndarray:
    model_outputs = model.apply(
        {"params": params},
        latents,
        timesteps,
    )
    if not isinstance(model_outputs, jnp.ndarray):
        raise ValueError("Model output is not a jnp.ndarray")
    losses = jnp.square(model_outputs - noise_values)
    return jnp.mean(losses)


@partial(jit, static_argnames=["model"])
def get_grads_and_loss(
    state: TrainState,
    latents: jnp.ndarray,
    noise_values: jnp.ndarray,
    timesteps: jnp.ndarray,
    model: UNet,
) -> Tuple[jnp.ndarray, Any]:
    def loss_fn(params: Any) -> jnp.ndarray:
        loss = get_loss(params, latents, noise_values, timesteps, model=model)
        return loss

    grad_fn = value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return loss, grads


def train_step(
    images: np.ndarray,
    state: TrainState,
    key: jnp.ndarray,
    model: UNet,
    alphas: jnp.ndarray,
) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
    key, key_t, key_n = jax.random.split(key, 3)
    latents, noise_values, timesteps = sample_latents(
        images, NUM_TIMESTEPS, alphas, key_t, key_n
    )
    loss, grads = get_grads_and_loss(state, latents, noise_values, timesteps, model)
    state = state.apply_gradients(grads=grads)
    return state, loss, key


@partial(jit, static_argnames=["model"])
def get_single_val_loss(
    images: np.ndarray,
    state: TrainState,
    key: jnp.ndarray,
    model: UNet,
    alphas: jnp.ndarray,
) -> jnp.ndarray:
    key, key_t, key_n = jax.random.split(key, 3)
    images = reshape_images(images)
    images = normalise_images(images)
    latents, noise_values, timesteps = sample_latents(
        images,
        NUM_TIMESTEPS,
        alphas,
        key_t,
        key_n,
    )
    return get_loss(state.params, latents, noise_values, timesteps, model=model)


def validate(
    val_generator: NumpyLoader,
    state: TrainState,
    model: UNet,
    alphas: jnp.ndarray,
    key: jnp.ndarray = PRNGKey(42),
) -> None:
    total_loss = 0
    total_samples = 0
    for images, _ in val_generator:
        key, _ = jax.random.split(key)
        loss = get_single_val_loss(images, state, key, model, alphas)
        total_loss += loss
        total_samples += images.shape[0]

    val_loss = total_loss / total_samples
    print(f"Validation loss: {val_loss * 1e3:.3f} * 10e-3")


def execute_train_loop(
    train_generator: NumpyLoader,
    val_generator: NumpyLoader,
    state: TrainState,
    model: UNet,
    epochs: int,
    expdir: Path,
    train_loss_every: int = -1,
    val_every: int = -1,
) -> TrainState:
    alphas = calculate_alphas(NUM_TIMESTEPS)
    key = PRNGKey(0)
    for epoch in range(epochs):
        print(f">>>>> Epoch {epoch} <<<<<")
        for step, (images, _) in enumerate(tqdm(train_generator)):
            images = reshape_images(images)
            images = normalise_images(images)
            state, loss, key = train_step(images, state, key, model, alphas)
            if train_loss_every > 0 and step % train_loss_every == 0:
                print(f"Training loss: {loss:.3f}")
            if val_every > 0 and step > 0 and step % val_every == 0:
                validate(val_generator, state, model, alphas)
        validate(val_generator, state, model, alphas)
        save_state(state, expdir / Path(f"model_parameters_epoch_{epoch}.pkl"))
    return state


def main(
    expdir: str = "",
    epochs: int = 10,
    train_loss_every: int = -1,
    val_every: int = -1,
    checkpoint_path: Optional[str] = None,
) -> None:
    if not expdir:
        raise ValueError("Please provide an experiment directory.")
    expdir_path = Path(expdir)
    train_generator, val_generator = get_dataset(BATCH_SIZE)
    model, parameters = initialize_model()
    print(f"Model has {count_params(parameters) / 10 ** 6:.1f} M parameters.")
    if checkpoint_path is not None:
        parameters = load_parameters(Path(checkpoint_path))
    state = TrainState.create(
        apply_fn=model.apply,
        params=parameters,
        tx=get_optimiser(),
    )
    state = execute_train_loop(
        train_generator,
        val_generator,
        state,
        model,
        epochs,
        expdir_path,
        train_loss_every,
        val_every,
    )


if __name__ == "__main__":
    fire.Fire(main)
