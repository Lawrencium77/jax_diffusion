"""
U-Net implementation. See https://arxiv.org/abs/1505.04597.
"""

from typing import Optional, Tuple
import jax.numpy as jnp
from flax import linen as nn
from jax.random import PRNGKey

from jax_diffusion.utils import NestedDict


class SinusoidalPositionalEmbeddings(nn.Module):
    d_model: int
    max_period: int = 10000

    @nn.compact
    def __call__(self, timesteps: jnp.ndarray) -> jnp.ndarray:
        """
        Sinusoidal embeddings as used in Attention is All You Need,
        """
        half_dim = self.d_model // 2

        emb_frequencies = jnp.log(self.max_period) / (half_dim - 1)
        emb_frequencies = jnp.exp(jnp.arange(half_dim) * -emb_frequencies)

        angle_rads = timesteps[:, None] * emb_frequencies[None, :]

        sin_embs = jnp.sin(angle_rads)
        cos_embs = jnp.cos(angle_rads)

        embeddings = jnp.concatenate([sin_embs, cos_embs], axis=-1)
        height = int(self.d_model**0.5)
        embeddings = embeddings.reshape(-1, height, height, 1)
        return embeddings


class ConvBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.swish(x)
        return x


class DownBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool, timesteps: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if timesteps is not None:
            x += SinusoidalPositionalEmbeddings(x.shape[1] ** 2)(timesteps)
        conv = ConvBlock(self.out_channels)(x, train)
        pooled = nn.max_pool(conv, window_shape=(2, 2), strides=(2, 2), padding="SAME")
        return conv, pooled


class UpBlock(nn.Module):
    out_channels: int

    @staticmethod
    def center_crop(tensor: jnp.ndarray, target_shape: Tuple[int]) -> jnp.ndarray:
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
        """
        Note that we crop the upsampled tensor, not the skip tensor.
        """
        if timesteps is not None:
            x += SinusoidalPositionalEmbeddings(x.shape[1] ** 2)(timesteps)
        upsampled = nn.ConvTranspose(
            features=self.out_channels, kernel_size=(2, 2), strides=(2, 2)
        )(x)
        if skip.shape[1:3] != upsampled.shape[1:3]:
            upsampled = self.center_crop(upsampled, skip.shape)

        concatenated = jnp.concatenate([upsampled, skip], axis=-1)
        return ConvBlock(self.out_channels)(concatenated, train)


class UNet(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, timesteps: jnp.ndarray, train: bool
    ) -> jnp.ndarray:
        conv1, pool1 = DownBlock(64)(x, train, timesteps)
        conv2, pool2 = DownBlock(128)(pool1, train, timesteps)
        conv3, pool3 = DownBlock(256)(
            pool2,
            train=train,
        )  # Non-even feat dim so don't apply timestep embeddings

        bottleneck = ConvBlock(512)(pool3, train)

        up3 = UpBlock(256)(bottleneck, conv3, train, timesteps)
        up2 = UpBlock(128)(
            up3,
            conv2,
            train,
        )  # Non-even feat dim so don't apply timestep embeddings
        up1 = UpBlock(64)(up2, conv1, train, timesteps)

        output = nn.Conv(self.out_channels, kernel_size=(1, 1), padding="SAME")(up1)
        return output


def initialize_model(
    key: PRNGKey, input_shape: Tuple[int] = (1, 28, 28, 1), num_classes: int = 1
) -> Tuple[UNet, NestedDict, NestedDict]:
    model = UNet(out_channels=num_classes)
    variables = model.init(
        key,
        jnp.ones(input_shape),
        jnp.ones(1),
        train=True,
    )
    return model, variables["params"], variables["batch_stats"]


if __name__ == "__main__":
    key = PRNGKey(0)
    model, variables = initialize_model(key)
    x = jnp.ones((128, 28, 28, 1))
    timesteps = jnp.arange(128)
    preds = model.apply(variables, x, timesteps)
    print("Input shape:", x.shape)
    print("Output shape:", preds.shape)
