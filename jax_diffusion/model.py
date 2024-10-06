from typing import Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from utils import ParamType


class PositionalEncoding(nn.Module):
    channels: int
    embedding_dim: int
    max_period: int = 10000

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class DoubleConv(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
class SelfAttention(nn.Module):
    head_size: int
    num_segments: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class DownBlock(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class UpBlock(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, residual: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError
    
class OutConv(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        raise NotImplementedError


class UNet(nn.Module):
    out_channels: int
    embedding_dim: int = 128
    num_groups: int = 32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, timesteps: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        # x has shape [bsz, 28, 28, 1]
        x1 = DoubleConv(64)(x)
        x2 = DownBlock(64, 128)(x1) + PositionalEncoding(128, 16)(timesteps)
        x3 = DownBlock(128, 256)(x2) + PositionalEncoding(256, 8)(timesteps)
        x3 = SelfAttention(256, 8)(x3)
        x4 = DownBlock(256, 512)(x3) + PositionalEncoding(256, 4)(timesteps)
        x4 = SelfAttention(256, 4)(x4)
        x = UpBlock(512, 256)(x4, x3) + PositionalEncoding(128, 8)(timesteps)
        x = SelfAttention(128, 8)(x)
        x = UpBlock(256, 128)(x, x2) + PositionalEncoding(64, 16)(timesteps)
        x = UpBlock(128, 64)(x, x1) + PositionalEncoding(64, 32)(timesteps)
        output = OutConv(64)(x)
        return output


def initialize_model(
    key: Array,
    input_shape: Tuple[int, ...] = (1, 28, 28, 1),
    num_classes: int = 1,
) -> Tuple[UNet, ParamType, ParamType]:
    model = UNet(out_channels=num_classes)
    variables = model.init(
        key,
        jnp.ones(input_shape),
        jnp.ones(1),
        train=True,
    )
    return model, variables["params"], {}
