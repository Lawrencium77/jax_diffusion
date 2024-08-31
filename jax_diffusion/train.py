from pathlib import Path
import jax.numpy as jnp
import numpy as np
import optax
import fire
from tqdm import tqdm
from flax.training import train_state

from dataset import NumpyLoader, get_dataset
from forward_process import sample_latents, calculate_alphas
from jax import value_and_grad, jit
from jax.random import PRNGKey
from model import initialize_model
from typing import Any, Optional, Tuple
from utils import (
    ParamType,
    count_params,
    normalise_images,
    reshape_images,
    save_model_parameters,
)

BATCH_SIZE = 128
NUM_TIMESTEPS = 1000


class TrainState(train_state.TrainState):
    batch_stats: Any


def get_optimiser(
    learning_rate: float = 0.0001,
) -> optax.GradientTransformation:
    return optax.adam(learning_rate)


def get_loss(
    params: ParamType,
    batch_stats: ParamType,
    latents: jnp.ndarray,
    noise_values: jnp.ndarray,
    timesteps: jnp.ndarray,
    train: bool,
) -> Tuple[jnp.ndarray, Optional[ParamType]]:
    """
    MSE loss with model application.
    """
    if train:
        model_outputs, updates = MODEL.apply(
            {"params": params, "batch_stats": batch_stats},
            latents,
            timesteps,
            train=train,
            mutable=["batch_stats"],
        )
    else:
        model_outputs = MODEL.apply(
            {"params": params, "batch_stats": batch_stats},
            latents,
            timesteps,
            train=train,
        )
        updates = None

    losses = jnp.square(model_outputs - noise_values)
    loss = jnp.mean(losses)
    return loss, updates


@jit
def get_grads_and_loss(
    state: TrainState,
    latents: jnp.ndarray,
    noise_values: jnp.ndarray,
    timesteps: jnp.ndarray,
) -> Tuple[jnp.ndarray, ParamType, Optional[ParamType]]:
    """
    Forward pass, backward pass, loss calculation.
    """

    def loss_fn(params: ParamType) -> Tuple[jnp.ndarray, Optional[ParamType]]:
        loss, updates = get_loss(
            params, state.batch_stats, latents, noise_values, timesteps, train=True
        )
        return loss, updates

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    return loss, grads, updates


def train_step(
    images: jnp.ndarray,
    state: TrainState,
) -> Tuple[TrainState, jnp.ndarray]:
    latents, noise_values, timesteps = sample_latents(images, NUM_TIMESTEPS, ALPHAS)
    loss, grads, updates = get_grads_and_loss(state, latents, noise_values, timesteps)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss


@jit
def get_single_val_loss(images: np.ndarray, state: TrainState) -> jnp.ndarray:
    images_normalised = normalise_images(images)
    images_reshaped = reshape_images(images_normalised)
    latents, noise_values, timesteps = sample_latents(
        images_reshaped, NUM_TIMESTEPS, ALPHAS
    )
    loss, _ = get_loss(
        state.params, state.batch_stats, latents, noise_values, timesteps, train=False
    )
    return loss


def validate(
    val_generator: NumpyLoader,
    state: TrainState,
) -> None:
    total_loss = 0
    total_samples = 0
    for images, _ in val_generator:
        loss = get_single_val_loss(images, state)
        total_loss += loss
        total_samples += images.shape[0]
    val_loss = total_loss / total_samples
    print(f"Validation loss: {val_loss * 1e3:.3f} * 10e-3")


def execute_train_loop(
    train_generator: NumpyLoader,
    val_generator: NumpyLoader,
    state: TrainState,
    epochs: int,
    expdir: Path,
    train_loss_every: int = -1,
    val_every: int = -1,
) -> TrainState:
    global ALPHAS
    ALPHAS = calculate_alphas(NUM_TIMESTEPS)
    for epoch in range(epochs):
        print(f">>>>> Epoch {epoch} <<<<<")
        for step, (images, _) in enumerate(tqdm(train_generator)):
            images = normalise_images(images)
            images = reshape_images(images)
            state, loss = train_step(images, state)
            if train_loss_every > 0 and step % train_loss_every == 0:
                print(f"Training loss: {loss:.3f}")
            if val_every > 0 and step > 0 and step % val_every == 0:
                validate(val_generator, state)
        validate(val_generator, state)
        save_model_parameters(
            state.params, expdir / Path(f"model_parameters_epoch_{epoch}.pkl")
        )
    return state


def main(
    expdir: Optional[str] = None,
    epochs: int = 10,
    train_loss_every: int = -1,
    val_every: int = -1,
) -> None:
    if expdir is None:
        raise ValueError("Please provide an experiment directory.")
    expdir_path = Path(expdir)
    global MODEL
    train_generator, val_generator = get_dataset(BATCH_SIZE)
    MODEL, parameters, batch_stats = initialize_model(PRNGKey(0))
    print(
        f"Initialised model with {count_params(parameters) / 10 ** 6:.1f} M parameters."
    )
    state = TrainState.create(
        apply_fn=MODEL.apply,
        params=parameters,
        tx=get_optimiser(),
        batch_stats=batch_stats,
    )
    state = execute_train_loop(
        train_generator,
        val_generator,
        state,
        epochs,
        expdir_path,
        train_loss_every,
        val_every,
    )


if __name__ == "__main__":
    fire.Fire(main)
