import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from dataset import NumpyLoader, get_dataset
from jax import value_and_grad, jit
from feedforward import init_all_parameters, forward_pass
from typing import List

BATCH_SIZE = 128
SIZES = [784, 128, 10]
EPSILON = 1e-8
IMAGE_NORMALISATION = 255.0

def normalise_images(images: np.ndarray) -> jnp.ndarray:
    return jnp.array(images, dtype=jnp.float32) / IMAGE_NORMALISATION

def get_optimiser(params, learning_rate=0.0001):
    optimiser = optax.adam(learning_rate)
    buffers = optimiser.init(params)
    return optimiser, buffers

def one_hot(labels: jnp.ndarray) -> jnp.ndarray:
    """
    One-hot encode labels.
    """
    return jnp.eye(10)[labels]

def ce_loss(
    params: jnp.ndarray, 
    images: jnp.ndarray, 
    labels: jnp.ndarray,
) -> jnp.ndarray:
    """
    Cross-entropy loss.
    """
    y_pred = forward_pass(params, images)  
    y_pred = jnp.clip(y_pred, EPSILON, 1.0 - EPSILON)  # Avoid log(0)
    losses = -1 * jnp.sum(labels * jnp.log(y_pred), axis=-1)  
    return jnp.mean(losses) 

def get_grads_and_loss(
    images: jnp.ndarray, 
    labels: jnp.ndarray, 
    params: List[List[jnp.ndarray]],
) -> jnp.ndarray:
    """
    Forward pass, backward pass, loss calculation.
    """
    return value_and_grad(ce_loss)(params, images, labels)

def training_step(
    images: jnp.ndarray, 
    labels: jnp.ndarray, 
    params: jnp.ndarray, 
    optimiser: optax._src.base.GradientTransformationExtraArgs, 
    buffers,
) -> jnp.ndarray:
    one_hot_labels = one_hot(labels)
    loss, grads = get_grads_and_loss(images, one_hot_labels, params)
    updates, buffers = optimiser.update(grads, buffers, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, buffers, loss

def execute_training_loop(
    training_generator: NumpyLoader,
    params: List[List[jnp.ndarray]],
    optimiser: optax._src.base.GradientTransformationExtraArgs,
    buffers,
    epochs: int = 10,
) -> List[List[jnp.ndarray]]:
    for epoch in range(epochs):
        print(f">>>>> Epoch {epoch} <<<<<")
        for images, labels in tqdm(training_generator): 
            images = normalise_images(images)
            params, buffers, loss = training_step(images, labels, params, optimiser, buffers)
        print(f"Train loss at end of epoch {epoch}: {loss}")
    return params

def main():
    _, training_generator = get_dataset(BATCH_SIZE)
    parameters = init_all_parameters(SIZES)
    optimiser, buffers = get_optimiser(parameters)
    parameters = execute_training_loop(training_generator, parameters, optimiser, buffers)

if __name__ == "__main__":
    main()
