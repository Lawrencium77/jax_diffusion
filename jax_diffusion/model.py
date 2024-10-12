from typing import Any, Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn
from jax import Array, image

from utils import SPATIAL_DIM, NUM_CHANNELS


class PositionalEncoding(nn.Module):
    channels: int
    embed_size: int
    max_period: int = 10000

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        batch_size = timesteps.shape[0]

        idx = jnp.arange(0, self.channels, 2)
        exponent = idx / self.channels
        inv_freq = 1.0 / (self.max_period**exponent)

        angle_rads = timesteps[:, None] * inv_freq[None, :]

        pos_enc_a = jnp.sin(angle_rads)
        pos_enc_b = jnp.cos(angle_rads)
        pos_enc = jnp.concatenate([pos_enc_a, pos_enc_b], axis=-1)

        pos_enc = pos_enc.reshape(batch_size, self.channels, 1, 1)
        pos_enc = jnp.tile(pos_enc, (1, 1, self.embed_size, self.embed_size))
        pos_enc = pos_enc.transpose((0, 2, 3, 1))
        return pos_enc


class DoubleConv(nn.Module):
    in_channels: int
    out_channels: int
    mid_channels: Optional[int] = None
    residual: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mid_channels = self.mid_channels or self.out_channels
        residual = x
        x = nn.Conv(
            features=mid_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.GroupNorm(num_groups=32)(x)
        x = nn.gelu(x)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)
        x = nn.GroupNorm(num_groups=32)(x)
        if self.residual:
            x = nn.gelu(x + residual)
        return x


class SelfAttention(nn.Module):
    feat_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch_size, seq_len, feat_dim]
        x_ln = nn.LayerNorm()(x)
        attention_value = nn.MultiHeadDotProductAttention(
            num_heads=4,
            qkv_features=self.feat_dim,
            out_features=self.feat_dim,
            dropout_rate=0.0,
            deterministic=True,
        )(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        y = nn.LayerNorm()(attention_value)
        y = nn.Dense(features=self.feat_dim)(y)
        y = nn.gelu(y)
        y = nn.Dense(features=self.feat_dim)(y)
        attention_output = y + attention_value
        return attention_output


class SAWrapper(nn.Module):
    num_channels: int
    image_width: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: [batch_size, height, width, channels] (JAX default)
        batch_size = x.shape[0]
        x = x.reshape(
            batch_size, self.image_width * self.image_width, self.num_channels
        )
        x = SelfAttention(feat_dim=self.num_channels)(x)
        x = x.reshape(batch_size, self.image_width, self.image_width, self.num_channels)
        return x


class DownBlock(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.max_pool(
            x, window_shape=(2, 2), strides=(2, 2), padding="SAME"
        )  # Requires shape [N, H, W, C]
        x = DoubleConv(self.in_channels, self.in_channels, residual=True)(x)
        x = DoubleConv(self.in_channels, self.out_channels)(x)
        return x


class UpBlock(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        x1 = image.resize(
            x1,
            shape=(
                x1.shape[0],
                x1.shape[1] * 2,
                x1.shape[2] * 2,
                x1.shape[3],
            ),
            method="bilinear",
        )
        x = jnp.concatenate([x1, x2], axis=-1)
        x = DoubleConv(self.in_channels, self.in_channels, residual=True)(x)
        x = DoubleConv(self.in_channels, self.out_channels, self.in_channels // 2)(x)
        return x


class OutConv(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(
            features=self.out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="SAME",
            use_bias=False,
        )(x)


class UNet(nn.Module):
    out_channels: int
    embedding_dim: int = 128

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, timesteps: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        # x has shape [bsz, 32, 32, 1]
        x1 = DoubleConv(1, 64)(x)  # [bsz, 32, 32, 64]
        x2 = DownBlock(64, 128)(x1) + PositionalEncoding(128, 16)(
            timesteps
        )  # [bsz, 16, 16, 128]
        x3 = DownBlock(128, 256)(x2) + PositionalEncoding(256, 8)(
            timesteps
        )  # [bsz, 8, 8, 256]
        x3 = SAWrapper(256, 8)(x3)
        x4 = DownBlock(256, 256)(x3) + PositionalEncoding(256, 4)(
            timesteps
        )  # [bsz, 4, 4, 256]
        x4 = SAWrapper(256, 4)(x4)
        x = UpBlock(512, 128)(x4, x3) + PositionalEncoding(128, 8)(
            timesteps
        )  # [bsz, 8, 8, 128]
        x = SAWrapper(128, 8)(x)
        x = UpBlock(256, 64)(x, x2) + PositionalEncoding(64, 16)(
            timesteps
        )  # [bsz, 16, 16, 64]
        x = UpBlock(128, 64)(x, x1) + PositionalEncoding(64, 32)(
            timesteps
        )  # [bsz, 32, 32, 64]
        output = OutConv(64, 1)(x)  # [bsz, 32, 32, 1]
        return output


def initialize_model(
    key: Array,
    input_shape: Tuple[int, ...] = (1, SPATIAL_DIM, SPATIAL_DIM, NUM_CHANNELS),
    num_classes: int = 1,
) -> Tuple[UNet, Any, Any]:
    model = UNet(out_channels=num_classes)
    variables = model.init(
        key,
        jnp.ones(input_shape),
        jnp.ones(1),
        train=True,
    )
    return model, variables["params"], {}
