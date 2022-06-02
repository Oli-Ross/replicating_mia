import pathlib
from os.path import abspath, dirname, join

import datasets
import numpy as np
import pytest


def test_dataset_files():
    datasetFiles = datasets.DatasetFiles("test")
    actualDataDir: str = abspath(datasetFiles.dataDirectory)
    currentDir: str = dirname(__file__)
    expectedDataDir: str = abspath(join(currentDir, "../../data/test"))

    assert actualDataDir == expectedDataDir

    expectedFeaturesFile = abspath(join(expectedDataDir, "features.npy"))
    actualFeaturesFile = abspath(datasetFiles.numpyFeatures)

    assert expectedFeaturesFile == actualFeaturesFile

    expectedLabelsFile = abspath(join(expectedDataDir, "labels.npy"))
    actualLabelsFile = abspath(datasetFiles.numpyLabels)

    assert expectedLabelsFile == actualLabelsFile


def test_baseclass():
    with pytest.raises(AssertionError):
        datasets.Dataset()


def test_kaggle():
    kaggle = datasets.KagglePurchaseDataset()

    assert kaggle.features.shape == (197324, 600)
    assert kaggle.labels.shape == (197324, 1)
    assert np.max(kaggle.features) != 0
    assert np.max(kaggle.labels) != 0


def test_cifar10():
    cifar10 = datasets.Cifar10Dataset()

    assert cifar10.features.shape == (60000, 32, 32, 3)
    assert cifar10.labels.shape == (60000, 1)
    assert np.max(cifar10.features) != 0
    assert np.max(cifar10.labels) != 0


def test_cifar100():
    cifar100 = datasets.Cifar100Dataset()

    assert cifar100.features.shape == (60000, 32, 32, 3)
    assert cifar100.labels.shape == (60000, 1)
    assert np.max(cifar100.features) != 0
    assert np.max(cifar100.labels) != 0
