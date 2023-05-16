"""
.. include:: ../docs/datasets.md
"""

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

from os.path import dirname, isdir, join
import shutil
from typing import Tuple

import numpy as np
import sklearn.cluster
import tensorflow as tf
from tensorflow.data import Dataset  # pyright: ignore
from tensorflow.keras.utils import to_categorical  # pyright: ignore
from tensorflow.python.framework import random_seed
from typing import List
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
    np.random.seed(global_seed)
    random_seed.set_seed(global_seed)


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
    print("Reading Kaggle from raw file.")
    rawDataFile: str = join(dataDir, "kaggle", "raw_data")
    data: NDArray = np.loadtxt(rawDataFile, dtype=int, delimiter=',')
    labels: NDArray = data[:, 0]
    features: NDArray = data[:, 1:]
    # 0-based index
    assert np.min(labels) >= 0
    labels = labels - np.min(labels)
    labels = to_categorical(labels, dtype='int64')
    return features, labels


def shuffle(dataset: Dataset) -> Dataset:
    datasetSize = dataset.cardinality().numpy()
    if datasetSize <= 0:  # tf uses constants < 0 to indicate unknown cardinality
        print("Warning: Getting dataset size from loading it to memory via numpy iterator, potentially slow.")
        datasetSize = len(list(dataset.as_numpy_iterator()))
    assert datasetSize > 0, "Dataset is empty"

    return dataset.shuffle(datasetSize, seed=global_seed, reshuffle_each_iteration=False)


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
    kmeans = sklearn.cluster.KMeans(n_clusters=numberOfClusters, random_state=global_seed)
    features, _ = _read_kaggle_data()
    kaggleSize = 197324
    labels: NDArray = kmeans.fit_predict(features).reshape(kaggleSize, 1)
    labels = to_categorical(labels, dtype='int64')
    return Dataset.from_tensor_slices((features, labels))


def load_attack(datasetName: str, verbose=True) -> Dataset:
    datasetDir: str = join(dataDir, "attack", datasetName, "dataset")
    if verbose:
        print(f"Loading dataset \"{datasetName}\" from disk.")
    return tf.data.experimental.load(datasetDir)


def load_shadow(datasetName: str, verbose=True) -> Dataset:
    datasetDir: str = join(dataDir, "shadow", datasetName, "dataset")
    if verbose:
        print(f"Loading dataset \"{datasetName}\" from disk.")
    return tf.data.experimental.load(datasetDir)


def save_attack(dataset: Dataset, datasetName: str):
    datasetDir: str = join(dataDir, "attack", datasetName, "dataset")
    tf.data.experimental.save(dataset, datasetDir)


def save_shadow(dataset: Dataset, datasetName: str):
    datasetDir: str = join(dataDir, "shadow", datasetName, "dataset")
    tf.data.experimental.save(dataset, datasetDir)


def delete_shadow(datasetName: str):
    datasetDir: str = join(dataDir, "shadow", datasetName, "dataset")
    shutil.rmtree(datasetDir)


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
    else:
        raise ValueError(f"{datasetName} is not a known dataset.")

    print(f"Saving {datasetName} to disk.")
    tf.data.experimental.save(dataset, datasetDir)
    return dataset


def split_dataset(dataset: Dataset, numSubsets: int) -> List[Dataset]:
    datasets = []
    for i in range(numSubsets):
        datasets.append(dataset.shard(numSubsets, i))
    return datasets


def load_all_datasets():
    load_dataset("cifar10")
    load_dataset("cifar100")
    load_dataset("kaggle")
    load_dataset("kaggle_2")
    load_dataset("kaggle_10")
    load_dataset("kaggle_20")
    load_dataset("kaggle_50")


if __name__ == "__main__":
    import argparse
    import configuration as con

    parser = argparse.ArgumentParser(description='Make sure the needed dataset is downloaded.')
    parser.add_argument('--config', help='Relative path to config file.',)
    config = con.from_cli_options(vars(parser.parse_args()))
    set_seed(config["seed"])

    dataName = config["targetDataset"]["name"]
    print(f"Loading {dataName}, doing clustering if necessary.")
    # Implicitely performs clustering for kaggle datasets
    load_dataset(dataName)
