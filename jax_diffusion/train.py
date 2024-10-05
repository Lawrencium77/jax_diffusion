from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import optax
import fire
from tqdm import tqdm

from dataset import NumpyLoader, get_dataset
from forward_process import sample_latents, calculate_alphas
from jax import value_and_grad, jit
from jax.random import PRNGKey
from model import initialize_model
from typing import Optional, Tuple
from utils import (
    ParamType,
    TrainState,
    count_params,
    load_state,
    normalise_images,
    reshape_images,
    save_state,
)

BATCH_SIZE = 128
NUM_TIMESTEPS = 1000


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
) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[ParamType]]:
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
    return loss, model_outputs, updates


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
        loss, _, updates = get_loss(
            params, state.batch_stats, latents, noise_values, timesteps, train=True
        )
        return loss, updates

    grad_fn = value_and_grad(loss_fn, has_aux=True)
    (loss, updates), grads = grad_fn(state.params)
    return loss, grads, updates


def train_step(
    images: jnp.ndarray,
    state: TrainState,
    rng_key: PRNGKey,
) -> Tuple[TrainState, jnp.ndarray, PRNGKey]:
    rng_key, key_t, key_n = jax.random.split(rng_key, 3)
    latents, noise_values, timesteps = sample_latents(
        images, NUM_TIMESTEPS, ALPHAS, key_t, key_n
    )
    loss, grads, updates = get_grads_and_loss(
        state, latents, noise_values, timesteps
    )
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss, rng_key


@jit
def get_single_val_loss(
    images: np.ndarray,
    state: TrainState,
    rng_key: PRNGKey,
) -> Tuple[
    jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray
]:
    rng_key, key_t, key_n = jax.random.split(rng_key, 3)
    images_normalised = normalise_images(images)
    images_reshaped = reshape_images(images_normalised)
    latents, noise_values, timesteps = sample_latents(
        images_reshaped, NUM_TIMESTEPS, ALPHAS, key_t, key_n
    )
    loss, model_outputs, _ = get_loss(
        state.params, state.batch_stats, latents, noise_values, timesteps, train=False
    )
    return loss, images_reshaped, latents, noise_values, timesteps, model_outputs


def validate(
    val_generator: NumpyLoader,
    state: TrainState,
    epoch: int,
    step: int,
    save_data: bool,
) -> None:
    total_loss = 0
    total_samples = 0
    rng_key = jax.random.PRNGKey(42)
    images_reshaped, latents, noise_values, timesteps, model_outputs = (
        None,
        None,
        None,
        None,
        None,
    )

    for images, _ in val_generator:
        rng_key, _ = jax.random.split(rng_key)
        loss, images_reshaped, latents, noise_values, timesteps, model_outputs = (
            get_single_val_loss(images, state, rng_key)
        )
        total_loss += loss # TODO: Does this need to be multipled by images.shape[0]?
        total_samples += images.shape[0]

    val_loss = total_loss / total_samples
    print(f"Validation loss: {val_loss * 1e3:.3f} * 10e-3")

    if save_data:
        if any(x is None for x in [images_reshaped, latents, noise_values, timesteps]):
            raise ValueError("Validation data were not computed correctly.")

        output_path = Path(f"validation_data_epoch_{epoch}_step_{step}.pkl")
        save_state(
            {
                "images": images_reshaped,
                "latents": latents,
                "noise_values": noise_values,
                "timesteps": timesteps,
                "model_outputs": model_outputs,
            },
            output_path,
        )


def execute_train_loop(
    train_generator: NumpyLoader,
    val_generator: NumpyLoader,
    state: TrainState,
    epochs: int,
    expdir: Path,
    save_val_outputs: bool,
    train_loss_every: int = -1,
    val_every: int = -1,
) -> TrainState:
    global ALPHAS
    ALPHAS = calculate_alphas(NUM_TIMESTEPS)
    rng_key = jax.random.PRNGKey(0)
    for epoch in range(epochs):
        print(f">>>>> Epoch {epoch} <<<<<")
        for step, (images, _) in enumerate(tqdm(train_generator)):
            images = normalise_images(images)
            images = reshape_images(images)
            state, loss, rng_key = train_step(images, state, rng_key)
            if train_loss_every > 0 and step % train_loss_every == 0:
                print(f"Training loss: {loss:.3f}")
            if val_every > 0 and step > 0 and step % val_every == 0:
                validate(
                    val_generator,
                    state,
                    epoch,
                    step,
                    save_val_outputs,
                )
        validate(
            val_generator,
            state,
            epoch,
            -1,
            save_val_outputs,
        )
        save_state(state, expdir / Path(f"model_parameters_epoch_{epoch}.pkl"))
    return state


def main(
    expdir: Optional[str] = None,
    epochs: int = 10,
    train_loss_every: int = -1,
    val_every: int = -1,
    save_val_outputs: bool = False,
    checkpoint_path: Optional[str] = None,
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
    if checkpoint_path is not None:
        chk_state = load_state(Path(checkpoint_path))
        parameters, batch_stats = chk_state["params"], chk_state["batch_stats"]
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
        save_val_outputs,
        train_loss_every,
        val_every,
    )


if __name__ == "__main__":
    fire.Fire(main)
