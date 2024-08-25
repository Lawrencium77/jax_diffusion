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

@jit
def get_grads_and_loss(
    images: jnp.ndarray, 
    labels: jnp.ndarray, 
    params: List[List[jnp.ndarray]],
) -> jnp.ndarray:
    """
    Forward pass, backward pass, loss calculation.
    """
    return value_and_grad(ce_loss)(params, images, labels)

def train_step(
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

@jit
def get_single_val_loss(images, labels, params):
    images = normalise_images(images)
    one_hot_labels = one_hot(labels)
    loss = ce_loss(params, images, one_hot_labels)
    return loss

def validate(
    val_generator: NumpyLoader, 
    params: List[List[jnp.ndarray]],
) -> float:
    total_loss = 0
    total_samples = 0
    for images, labels in val_generator:
        loss = get_single_val_loss(images, labels, params)
        total_loss += loss
        total_samples += len(labels)
    return total_loss / total_samples

def execute_train_loop(
    train_generator: NumpyLoader,
    val_generator: NumpyLoader,
    params: List[List[jnp.ndarray]],
    optimiser: optax._src.base.GradientTransformationExtraArgs,
    buffers,
    epochs: int = 10,
) -> List[List[jnp.ndarray]]:
    for epoch in range(epochs):
        print(f">>>>> Epoch {epoch} <<<<<")
        for images, labels in tqdm(train_generator): 
            images = normalise_images(images)
            params, buffers, _ = train_step(images, labels, params, optimiser, buffers)
        val_loss = validate(val_generator, params)
        print(f"Validation loss: {val_loss * 1e3:.2f} * 10e-3")
    return params

def main():
    _, train_generator, val_generator = get_dataset(BATCH_SIZE)
    parameters = init_all_parameters(SIZES)
    optimiser, buffers = get_optimiser(parameters)
    parameters = execute_train_loop(train_generator, val_generator, parameters, optimiser, buffers)

if __name__ == "__main__":
    main()
