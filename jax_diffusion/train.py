from pathlib import Path
import jax.numpy as jnp
import optax
import fire
from tqdm import tqdm
from flax.training import train_state  # Import train_state from Flax

from dataset import NumpyLoader, get_dataset
from forward_process import sample_latents, calculate_alphas
from jax import value_and_grad, jit
from jax.random import PRNGKey
from model import initialize_model
from typing import Tuple
from utils import (
    count_parameters,
    normalise_images,
    reshape_images,
    save_model_parameters,
)

BATCH_SIZE = 128
NUM_TIMESTEPS = 1000


def get_optimiser(learning_rate=0.0001):
    return optax.adam(learning_rate)


def get_loss(
    params: jnp.ndarray,
    latents: jnp.ndarray,
    noise_values: jnp.ndarray,
    timesteps: jnp.ndarray,
    train: bool,
) -> jnp.ndarray:
    """
    MSE loss.
    """
    y_pred = MODEL.apply(
        params,
        latents,
        timesteps,
        train,
    )
    losses = jnp.square(y_pred - noise_values)
    return jnp.mean(losses)


@jit
def get_grads_and_loss(
    state: train_state.TrainState,
    latents: jnp.ndarray,
    noise_values: jnp.ndarray,
    timesteps: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Forward pass, backward pass, loss calculation.
    """

    def loss_fn(params):
        return get_loss(params, latents, noise_values, timesteps, train=True)

    loss, grads = value_and_grad(loss_fn)(state.params)
    return loss, grads


def train_step(
    images: jnp.ndarray,
    state: train_state.TrainState,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    latents, noise_values, timesteps = sample_latents(images, NUM_TIMESTEPS, ALPHAS)
    loss, grads = get_grads_and_loss(state, latents, noise_values, timesteps)
    state = state.apply_gradients(grads=grads)
    return state, loss


@jit
def get_single_val_loss(images, params):
    images = normalise_images(images)
    images = reshape_images(images)
    latents, noise_values, timesteps = sample_latents(images, NUM_TIMESTEPS, ALPHAS)
    loss = get_loss(params, latents, noise_values, timesteps, train=False)
    return loss


def validate(
    val_generator: NumpyLoader,
    params: jnp.ndarray,
) -> float:
    total_loss = 0
    total_samples = 0
    for images, _ in val_generator:
        loss = get_single_val_loss(images, params)
        total_loss += loss
        total_samples += images.shape[0]
    return total_loss / total_samples


def execute_train_loop(
    train_generator: NumpyLoader,
    val_generator: NumpyLoader,
    state: train_state.TrainState,
    epochs: int,
    print_train_loss: bool,
) -> train_state.TrainState:
    global ALPHAS
    ALPHAS = calculate_alphas(NUM_TIMESTEPS)
    for epoch in range(epochs):
        print(f">>>>> Epoch {epoch} <<<<<")
        for images, _ in tqdm(train_generator):
            images = normalise_images(images)
            images = reshape_images(images)
            state, loss = train_step(images, state)
            if print_train_loss:
                print(f"Training loss: {loss:.3f}")
        val_loss = validate(val_generator, state.params)
        print(f"Validation loss: {val_loss * 1e3:.3f} * 10e-3")
    return state


def main(
    expdir: str = None,
    epochs: int = 10,
    print_train_loss: bool = False,
):
    if expdir is None:
        raise ValueError("Please provide an experiment directory.")
    expdir = Path(expdir)
    global MODEL
    train_generator, val_generator = get_dataset(BATCH_SIZE)
    MODEL, parameters = initialize_model(PRNGKey(0))
    print(
        f"Initialised model with {count_parameters(parameters) / 10 ** 6:.1f} M parameters."
    )
    state = train_state.TrainState.create(
        apply_fn=MODEL.apply,
        params=parameters,
        tx=get_optimiser(),
    )
    state = execute_train_loop(
        train_generator,
        val_generator,
        state,
        epochs,
        print_train_loss,
    )
    save_model_parameters(state.params, expdir / "model_parameters.pkl")


if __name__ == "__main__":
    fire.Fire(main)
