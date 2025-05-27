from pathlib import Path

import pytest
import torch

from src.data.mnist_datamodule import MNISTDataModule
from src.data.celeba_datamodule import CelebADataModule


@pytest.mark.parametrize("batch_size", [32, 128])
def test_mnist_datamodule(batch_size: int) -> None:
    """Tests `MNISTDataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = MNISTDataModule(data_dir=data_dir, batch_size=batch_size)
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "MNIST").exists()
    assert Path(data_dir, "MNIST", "raw").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    assert num_datapoints == 70_000

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64


@pytest.mark.parametrize("batch_size", [32, 64])
def test_celeba_datamodule(batch_size: int) -> None:
    """Tests `CelebADataModule` to verify that it can be downloaded correctly, that the necessary
    attributes were created (e.g., the dataloader objects), and that dtypes and batch sizes
    correctly match.

    :param batch_size: Batch size of the data to be loaded by the dataloader.
    """
    data_dir = "data/"

    dm = CelebADataModule(data_dir=data_dir, batch_size=batch_size)
    # dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, "celeba").exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    # Check dataset sizes
    assert len(dm.data_train) == 162_770  # CelebA default train split
    assert len(dm.data_val) == 19_867     # CelebA default validation split
    assert len(dm.data_test) == 19_962    # CelebA default test split

    # Check a batch
    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.int64  # CelebA attributes are binary, stored as float32
    assert x.shape[1] == 3  # RGB images
    assert x.shape[2] == 64  # Default image size
    assert x.shape[3] == 64  # Default image size
    assert y.shape[1] == 40  # 40 binary attributes
