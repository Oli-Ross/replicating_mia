from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

from os.path import abspath, dirname, join

import tensorflow as tf
import datasets
import numpy as np
import pytest


class TestDatasetFiles():

    def test_all(self):
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


class TestDatasetBasic():
    def test_baseclass(self):
        with pytest.raises(AssertionError):
            datasets.DatasetBaseClass()

    def test_kaggle(self):
        kaggle = datasets.KagglePurchaseDataset()

        assert kaggle.features.shape == (197324, 600)
        assert kaggle.labels.shape == (197324, 1)
        assert np.max(kaggle.features) != 0
        assert np.max(kaggle.labels) != 0

    def test_cifar10(self):
        cifar10 = datasets.Cifar10Dataset()

        assert cifar10.features.shape == (60000, 32, 32, 3)
        assert cifar10.labels.shape == (60000, 1)
        assert np.max(cifar10.features) != 0
        assert np.max(cifar10.labels) != 0

    def test_cifar100(self):
        cifar100 = datasets.Cifar100Dataset()

        assert cifar100.features.shape == (60000, 32, 32, 3)
        assert cifar100.labels.shape == (60000, 1)
        assert np.max(cifar100.features) != 0
        assert np.max(cifar100.labels) != 0

    @pytest.mark.skip("Takes too long.")
    def test_kaggle_cluster(self):
        kaggle_clustered_10 = datasets.KagglePurchaseDatasetClustered(10)

        labels = kaggle_clustered_10.labels
        assert labels.min() == 0
        assert labels.max() == 9

        kaggle_clustered_100 = datasets.KagglePurchaseDatasetClustered(100)

        labels = kaggle_clustered_100.labels
        assert labels.min() == 0
        assert labels.max() == 99


class TestDatasetSplit():

    def test_split_automatic(self):
        cifar10 = datasets.Cifar10Dataset()
        (x_train, y_train), (x_test, y_test) = cifar10.split()
        assert x_train.shape[0] == cifar10.train_size
        assert y_train.shape[0] == cifar10.train_size
        assert x_test.shape[0] == cifar10.size - cifar10.train_size
        assert y_test.shape[0] == cifar10.size - cifar10.train_size

    @pytest.mark.skip("Takes too long.")
    def test_random_split(self):
        kaggle = datasets.KagglePurchaseDataset()
        (x_train, y_train), (x_test, y_test) = kaggle.split_random()

        assert x_train.shape == kaggle.features[0:10000].shape
        with pytest.raises(AssertionError):
            np.testing.assert_equal(x_train, kaggle.features[0:10000])

        assert x_test.shape == kaggle.features[10000:197324].shape
        with pytest.raises(AssertionError):
            np.testing.assert_equal(x_test, kaggle.features[10000:197324])

        assert y_train.shape == kaggle.labels[0:10000].shape
        with pytest.raises(AssertionError):
            np.testing.assert_equal(y_train, kaggle.labels[0:10000])

        assert y_test.shape == kaggle.labels[10000:197324].shape
        with pytest.raises(AssertionError):
            np.testing.assert_equal(y_test, kaggle.labels[10000:197324])

    @pytest.mark.skip("Takes too long.")
    def test_cifar_split(self):
        cifar10 = datasets.Cifar10Dataset()

        (x_train, y_train), (x_test, y_test) = cifar10.split()
        (x_train_tf, y_train_tf), (x_test_tf,
                                   y_test_tf) = tf.keras.datasets.cifar10.load_data()

        # Assert that the dataset class correctly constructs the splits,
        # identical to the call to tensorflow
        np.testing.assert_equal(x_train, x_train_tf)
        np.testing.assert_equal(y_train, y_train_tf)
        np.testing.assert_equal(x_test, x_test_tf)
        np.testing.assert_equal(y_test, y_test_tf)

    def test_custom_split_fail(self):
        kaggle = datasets.KagglePurchaseDataset()

        with pytest.raises(ValueError):
            # train_size > dataset size
            kaggle.split(train_size=200000)
