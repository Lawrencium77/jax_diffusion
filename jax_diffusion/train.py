import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from dataset import NumpyLoader, get_dataset
from diffusion_utils import sample_latents, calculate_alphas, normalise_images
from jax import value_and_grad, jit
from feedforward import init_all_parameters, forward_pass
from typing import List, Tuple

BATCH_SIZE = 128
SIZES = [784, 784, 784]
NUM_TIMESTEPS = 1000

def get_optimiser(params, learning_rate=0.0001):
    optimiser = optax.adam(learning_rate)
    buffers = optimiser.init(params)
    return optimiser, buffers

def get_loss(
    params: jnp.ndarray, 
    latents: jnp.ndarray, 
    noise_values: jnp.ndarray,
    timesteps: jnp.ndarray,
) -> jnp.ndarray:
    """
    MSE loss.
    """
    y_pred = forward_pass(params, latents, timesteps)  
    losses = jnp.square(y_pred - noise_values) 
    return jnp.mean(losses) 

@jit
def get_grads_and_loss(
    params: List[List[jnp.ndarray]],
    latents: jnp.ndarray, 
    noise_values: jnp.ndarray,
    timesteps: jnp.ndarray,
) -> jnp.ndarray:
    """
    Forward pass, backward pass, loss calculation.
    """
    return value_and_grad(get_loss)(params, latents, noise_values, timesteps)

def train_step(
    images: jnp.ndarray, 
    params: List[List[jnp.ndarray]], 
    optimiser: optax._src.base.GradientTransformationExtraArgs, 
    buffers,
) -> Tuple[List[List[jnp.ndarray]], jnp.ndarray, jnp.ndarray]:
    latents, noise_values, timesteps = sample_latents(images, NUM_TIMESTEPS, ALPHAS)
    loss, grads = get_grads_and_loss(params, latents, noise_values, timesteps)
    updates, buffers = optimiser.update(grads, buffers, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, buffers, loss

@jit
def get_single_val_loss(images, params):
    images = normalise_images(images)
    latents, noise_values, timesteps = sample_latents(images, NUM_TIMESTEPS, ALPHAS)
    loss = get_loss(params, latents, noise_values, timesteps)
    return loss

def validate(
    val_generator: NumpyLoader, 
    params: List[List[jnp.ndarray]],
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
    params: List[List[jnp.ndarray]],
    optimiser: optax._src.base.GradientTransformationExtraArgs,
    buffers,
    epochs: int = 10,
) -> List[List[jnp.ndarray]]:
    global ALPHAS
    ALPHAS = calculate_alphas(train_generator, NUM_TIMESTEPS)
    for epoch in range(epochs):
        print(f">>>>> Epoch {epoch} <<<<<")
        for images, _ in tqdm(train_generator): 
            images = normalise_images(images)
            params, buffers, _ = train_step(images, params, optimiser, buffers)
        val_loss = validate(val_generator, params)
        print(f"Validation loss: {val_loss * 1e3:.2f} * 10e-3")
    return params

def main():
    train_generator, val_generator = get_dataset(BATCH_SIZE)
    parameters = init_all_parameters(SIZES)
    optimiser, buffers = get_optimiser(parameters)
    parameters = execute_train_loop(
        train_generator, 
        val_generator, 
        parameters, 
        optimiser, 
        buffers,
    )

if __name__ == "__main__":
    main()
