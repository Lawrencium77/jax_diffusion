"""
Modified Implementation of U-Net model to ensure output shape matches input shape
"""

import jax.numpy as jnp
from flax import linen as nn
from jax.random import PRNGKey


class ConvBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.out_channels, kernel_size=(3, 3), padding="SAME")(x)
        x = nn.relu(x)
        return x


class DownBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x):
        conv = ConvBlock(self.out_channels)(x)
        pooled = nn.max_pool(conv, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        return conv, pooled


class UpBlock(nn.Module):
    out_channels: int

    @nn.compact
    def __call__(self, x, skip):
        upsampled = nn.ConvTranspose(features=self.out_channels, kernel_size=(2, 2), strides=(2, 2))(x)
        
        def center_crop(tensor, target_shape):
            """Crop the center of the tensor to the target_shape."""
            diff_height = tensor.shape[1] - target_shape[1]
            diff_width = tensor.shape[2] - target_shape[2]
            crop_h = diff_height // 2
            crop_w = diff_width // 2
            return tensor[:, crop_h:crop_h + target_shape[1], crop_w:crop_w + target_shape[2], :]

        if skip.shape[1:3] != upsampled.shape[1:3]:
            skip = center_crop(skip, upsampled.shape)
        
        concatenated = jnp.concatenate([upsampled, skip], axis=-1)
        return ConvBlock(self.out_channels)(concatenated)


class UNet(nn.Module):
    out_channels: int  

    @nn.compact
    def __call__(self, x):
        conv1, pool1 = DownBlock(64)(x)
        conv2, pool2 = DownBlock(128)(pool1)
        conv3, pool3 = DownBlock(256)(pool2)

        bottleneck = ConvBlock(512)(pool3)

        up3 = UpBlock(256)(bottleneck, conv3)
        up2 = UpBlock(128)(up3, conv2)
        up1 = UpBlock(64)(up2, conv1)

        output = nn.Conv(self.out_channels, kernel_size=(1, 1), padding="SAME")(up1)
        return output  


def initialize_model(key, input_shape=(1, 28, 28, 1), num_classes=1):
    model = UNet(out_channels=num_classes)
    variables = model.init(key, jnp.ones(input_shape))
    return model, variables


if __name__ == "__main__":
    key = PRNGKey(0)
    model, variables = initialize_model(key)
    x = jnp.ones((1, 28, 28, 1))  
    preds = model.apply(variables, x)
    print("Input shape:", x.shape)
    print("Output shape:", preds.shape)
