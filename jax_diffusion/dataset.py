from typing import Tuple
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST
from torch.utils.data import random_split


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


def get_dataset(
    batch_size: int,
    val_split: float = 0.2,
) -> Tuple[MNIST, NumpyLoader, NumpyLoader]:
    mnist_dataset = MNIST("/tmp/mnist/", download=True, transform=FlattenAndCast())

    val_size = int(len(mnist_dataset) * val_split)
    train_size = len(mnist_dataset) - val_size

    train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

    train_loader = NumpyLoader(
        train_dataset, batch_size=batch_size, num_workers=0, shuffle=True
    )
    val_loader = NumpyLoader(
        val_dataset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    return train_loader, val_loader
