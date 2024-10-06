from typing import Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from utils import ParamType


class TimeEmbedding(nn.Module):
    embedding_dim: int

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        half_dim = self.embedding_dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

        emb = nn.Dense(self.embedding_dim * 4)(emb)
        emb = nn.swish(emb)
        emb = nn.Dense(self.embedding_dim)(emb)
        return emb
    
class AttentionBlock(nn.Module):
    num_heads: int
    num_groups: int = 32
    embedding_dim: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch, height, width, channels = x.shape
        h = nn.GroupNorm(num_groups=self.num_groups)(x)
        h = h.reshape(batch, height * width, channels)
        h = nn.SelfAttention(
            num_heads=self.num_heads,
            qkv_features=self.embedding_dim if self.embedding_dim else channels,
            out_features=channels,
            use_bias=True,
            dtype=x.dtype,
        )(h)

        h = h.reshape(batch, height, width, channels)
        return x + h


class ConvBlock(nn.Module):
    out_channels: int
    num_groups: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: Optional[jnp.ndarray], train: bool) -> jnp.ndarray:
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        if emb is not None:
            emb_out = nn.Dense(self.out_channels)(emb)
            emb_out = emb_out[:, None, None, :]  # Broadcast to match spatial dimensions
            h = h + emb_out
        h = nn.swish(h)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME")(h)
        h = nn.GroupNorm(num_groups=self.num_groups)(h)
        if emb is not None:
            emb_out = nn.Dense(self.out_channels)(emb)
            emb_out = emb_out[:, None, None, :]
            h = h + emb_out
        h = nn.swish(h)
        return h
class ResAttnBlock(nn.Module):
    out_channels: int
    embedding_dim: int = 128
    num_heads: int = 8
    num_groups: int = 32
    add_attention: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: Optional[jnp.ndarray], train: bool) -> jnp.ndarray:
        h = ConvBlock(self.out_channels, self.num_groups)(x, emb, train)
        if self.add_attention:
            h = AttentionBlock(
                num_heads=self.num_heads,
                num_groups=self.num_groups,
                embedding_dim=self.embedding_dim,
            )(h)
        return h


class DownBlock(nn.Module):
    out_channels: int
    num_groups: int = 32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool, timesteps: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        emb = None
        if timesteps is not None:
            emb = TimeEmbedding(self.embedding_dim)(timesteps)
            emb = nn.Dense(self.embedding_dim)(emb)
            emb = nn.swish(emb)
        conv = ResAttnBlock(self.out_channels, self.num_groups)(x, emb, train)
        pooled = nn.max_pool(conv, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        return conv, pooled


class UpBlock(nn.Module):
    out_channels: int
    embedding_dim: int = 128
    num_groups: int = 32

    @staticmethod
    def center_crop(tensor: jnp.ndarray, target_shape: Tuple[int, ...]) -> jnp.ndarray:
        """
        Crop the center of the tensor to the target_shape.
        """
        diff_height = tensor.shape[1] - target_shape[1]
        diff_width = tensor.shape[2] - target_shape[2]
        crop_h = diff_height // 2
        crop_w = diff_width // 2
        return tensor[
            :, crop_h : crop_h + target_shape[1], crop_w : crop_w + target_shape[2], :
        ]

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        skip: jnp.ndarray,
        train: bool,
        timesteps: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        emb = None
        if timesteps is not None:
            emb = TimeEmbedding(self.embedding_dim)(timesteps)
            emb = nn.Dense(self.embedding_dim)(emb)
            emb = nn.swish(emb)
        upsampled = nn.ConvTranspose(
            features=self.out_channels, kernel_size=(2, 2), strides=(2, 2)
        )(x)
        if skip.shape[1:3] != upsampled.shape[1:3]:
            upsampled = self.center_crop(upsampled, skip.shape)

        concatenated = jnp.concatenate([upsampled, skip], axis=-1)
        return ResAttnBlock(self.out_channels, self.num_groups)(concatenated, emb, train)


class UNet(nn.Module):
    out_channels: int
    embedding_dim: int = 128
    num_groups: int = 32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, timesteps: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        conv1, pool1 = DownBlock(64, self.num_groups)(x, train, timesteps)
        conv2, pool2 = DownBlock(128, self.num_groups)(pool1, train, timesteps)
        conv3, pool3 = DownBlock(256, self.num_groups)(pool2, train, timesteps)

        emb = TimeEmbedding(self.embedding_dim)(timesteps)
        emb = nn.Dense(self.embedding_dim)(emb)
        emb = nn.swish(emb)
        bottleneck = ConvBlock(512, self.num_groups)(pool3, emb, train)

        up3 = UpBlock(256, self.embedding_dim, self.num_groups)(bottleneck, conv3, train, timesteps)
        up2 = UpBlock(128, self.embedding_dim, self.num_groups)(up3, conv2, train, timesteps)
        up1 = UpBlock(64, self.embedding_dim, self.num_groups)(up2, conv1, train, timesteps)

        output = nn.Conv(self.out_channels, kernel_size=(1, 1), padding="SAME")(up1)
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
