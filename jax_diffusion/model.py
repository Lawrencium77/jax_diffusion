from typing import Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn
from jax import Array

from utils import ParamType


class SinusoidalPositionalEmbeddings(nn.Module):
    embedding_dim: int
    max_period: int = 10000

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        """
        Compute sinusoidal embeddings for timesteps.
        """
        half_dim = self.embedding_dim // 2
        emb_frequencies = jnp.exp(
            -jnp.log(self.max_period) * jnp.arange(half_dim) / (half_dim - 1)
        )
        angle_rads = timesteps[:, None] * emb_frequencies[None, :]
        embeddings = jnp.concatenate([jnp.sin(angle_rads), jnp.cos(angle_rads)], axis=-1)
        return embeddings


class ConvBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, emb: Optional[jnp.ndarray], train: bool) -> jnp.ndarray:
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        h = nn.BatchNorm(use_running_average=not train)(h)
        if emb is not None:
            emb_out = nn.Dense(self.out_channels)(emb)
            emb_out = emb_out[:, None, None, :]  # Broadcast to match spatial dimensions
            h = h + emb_out
        h = nn.swish(h)
        h = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME")(h)
        h = nn.BatchNorm(use_running_average=not train)(h)
        if emb is not None:
            emb_out = nn.Dense(self.out_channels)(emb)
            emb_out = emb_out[:, None, None, :]
            h = h + emb_out
        h = nn.swish(h)
        return h


class DownBlock(nn.Module):
    out_channels: int
    embedding_dim: int = 128

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool, timesteps: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        emb = None
        if timesteps is not None:
            emb = SinusoidalPositionalEmbeddings(self.embedding_dim)(timesteps)
            emb = nn.Dense(self.embedding_dim)(emb)
            emb = nn.swish(emb)
        conv = ConvBlock(self.out_channels)(x, emb, train)
        pooled = nn.max_pool(conv, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        return conv, pooled


class UpBlock(nn.Module):
    out_channels: int
    embedding_dim: int = 128

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
            emb = SinusoidalPositionalEmbeddings(self.embedding_dim)(timesteps)
            emb = nn.Dense(self.embedding_dim)(emb)
            emb = nn.swish(emb)
        upsampled = nn.ConvTranspose(
            features=self.out_channels, kernel_size=(2, 2), strides=(2, 2)
        )(x)
        if skip.shape[1:3] != upsampled.shape[1:3]:
            upsampled = self.center_crop(upsampled, skip.shape)

        concatenated = jnp.concatenate([upsampled, skip], axis=-1)
        return ConvBlock(self.out_channels)(concatenated, emb, train)


class UNet(nn.Module):
    out_channels: int
    embedding_dim: int = 128

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, timesteps: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        conv1, pool1 = DownBlock(64, self.embedding_dim)(x, train, timesteps)
        conv2, pool2 = DownBlock(128, self.embedding_dim)(pool1, train, timesteps)
        conv3, pool3 = DownBlock(256, self.embedding_dim)(pool2, train, timesteps)

        emb = SinusoidalPositionalEmbeddings(self.embedding_dim)(timesteps)
        emb = nn.Dense(self.embedding_dim)(emb)
        emb = nn.swish(emb)
        bottleneck = ConvBlock(512)(pool3, emb, train)

        up3 = UpBlock(256, self.embedding_dim)(bottleneck, conv3, train, timesteps)
        up2 = UpBlock(128, self.embedding_dim)(up3, conv2, train, timesteps)
        up1 = UpBlock(64, self.embedding_dim)(up2, conv1, train, timesteps)

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
    return model, variables["params"], variables["batch_stats"]
