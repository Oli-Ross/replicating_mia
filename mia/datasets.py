"""
.. include:: ../docs/datasets.md
"""

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import csv
from os.path import dirname, isdir, join
from typing import Tuple

import numpy as np
import sklearn.cluster
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical
from numpy.typing import NDArray

dataDir = join(dirname(__file__), "../data")
global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed


def _dataset_from_split(
        x_train, y_train, x_test, y_test) -> Dataset:
    """
    Using the provided split dataset, create a Dataset.
    """
    features: NDArray = np.append(x_train, x_test, axis=0).astype(np.float64)
    labels: NDArray = np.append(y_train, y_test, axis=0).astype(np.int32)
    labels = to_categorical(labels)
    return Dataset.from_tensor_slices((features, labels))


def _prepare_cifar100() -> Dataset:
    train, test = tf.keras.datasets.cifar100.load_data()
    return _dataset_from_split(train[0], train[1], test[0], test[1])


def _prepare_cifar10() -> Dataset:
    train, test = tf.keras.datasets.cifar10.load_data()
    return _dataset_from_split(train[0], train[1], test[0], test[1])


def _read_kaggle_data() -> Tuple[NDArray, NDArray]:
    """
    Read the Kaggle dataset features and labels from disk into Numpy arrays.
    """
    rawData: str = join(dataDir, "kaggle", "raw_data")
    kaggleSize = 197324
    labels: NDArray = np.zeros([kaggleSize, 1])
    features: NDArray = np.zeros([kaggleSize, 600])
    with open(rawData) as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            labels[index, 0] = row[0]
            features[index, :] = row[1:]
    labels = to_categorical(labels)
    return features, labels


def shuffle_kaggle(kaggle: Dataset) -> Dataset:
    """
    Shuffles Kaggle Dataset and datasets derived from it via clustering.
    """
    kaggleSize = 197324
    return kaggle.shuffle(kaggleSize, seed=global_seed,
                          reshuffle_each_iteration=False)


def _prepare_kaggle() -> Dataset:
    """
    Create Kaggle as Dataset from Numpy arrays
    """
    features, labels = _read_kaggle_data()
    return Dataset.from_tensor_slices((features, labels))


def _prepare_clustered_kaggle(numberOfClusters: int):
    """
    Load the Kaggle data and cluster it.
    """
    print(f"Clustering Kaggle with {numberOfClusters} classes.")
    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=numberOfClusters,
        random_state=global_seed)
    features, _ = _read_kaggle_data()
    kaggleSize = 197324
    labels: NDArray = kmeans.fit_predict(features).reshape(kaggleSize, 1)
    return Dataset.from_tensor_slices((features, labels))


def load_dataset(datasetName: str) -> Dataset:
    """
    Load a dataset.

    Valid `datasetName` values are: "cifar10", "cifar100", "kaggle", "kaggle_2",
    "kaggle_10","kaggle_20","kaggle_50","kaggle_100".
    """
    datasetDir: str = join(dataDir, datasetName, "dataset")
    if isdir(datasetDir):
        print(f"Loading {datasetName} from disk.")
        return tf.data.experimental.load(datasetDir)

    print(f"Loading {datasetName}.")

    if datasetName == "cifar10":
        dataset = _prepare_cifar10()
    elif datasetName == "cifar100":
        dataset = _prepare_cifar100()
    elif datasetName == "kaggle":
        dataset = _prepare_kaggle()
    elif datasetName == "kaggle_2":
        dataset = _prepare_clustered_kaggle(2)
    elif datasetName == "kaggle_10":
        dataset = _prepare_clustered_kaggle(10)
    elif datasetName == "kaggle_20":
        dataset = _prepare_clustered_kaggle(20)
    elif datasetName == "kaggle_50":
        dataset = _prepare_clustered_kaggle(50)
    elif datasetName == "kaggle_100":
        dataset = _prepare_clustered_kaggle(100)
    else:
        raise ValueError(f"{datasetName} is not a known dataset.")

    print(f"Saving {datasetName} to disk.")
    tf.data.experimental.save(dataset, datasetDir)
    return dataset


def load_all_datasets():
    load_dataset("cifar10")
    load_dataset("cifar100")
    load_dataset("kaggle")
    load_dataset("kaggle_2")
    load_dataset("kaggle_10")
    load_dataset("kaggle_20")
    load_dataset("kaggle_50")
    load_dataset("kaggle_100")


if __name__ == "__main__":
    load_all_datasets()
