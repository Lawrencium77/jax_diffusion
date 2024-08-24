"""
Super-simple feedforward neural net.
"""
import jax.numpy as jnp

from jax import nn, random
from typing import Callable, List

def init_single_layer_params(
        weights_initializer: Callable,
        biases_initializer: Callable,
        input_dim: int, 
        output_dim: int,
) -> List[jnp.ndarray]:
    """
    Initialise the parameters for a single matmul
    """
    weights = weights_initializer(random.key(42), (input_dim, output_dim))
    biases = biases_initializer(random.key(42), (output_dim,))  
    return [weights, biases]

def init_all_parameters(dims: List[int]) -> List[List[jnp.ndarray]]:
    """
    Initialise all parameters for a feedforward neural network
    dims specifies the feature dim at each point in the network
    """
    weights_initializer = nn.initializers.glorot_normal()
    biases_initializer = nn.initializers.zeros
    return [init_single_layer_params(
        weights_initializer,
        biases_initializer,
        dims[i],
        dims[i+1],
    ) for i in range(len(dims)-1)]

def forward_pass(params: List[List[jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    """
    Forward pass of a feedforward neural network
    """
    for layer_params in params:
        weights, biases = layer_params
        x = jnp.matmul(x, weights) + biases
        x = nn.gelu(x)
    return nn.softmax(x)