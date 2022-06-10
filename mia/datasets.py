from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import csv
from os.path import dirname, isdir, join
from typing import Tuple

import numpy as np
import sklearn.cluster
import tensorflow as tf
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
        x_train, y_train, x_test, y_test) -> tf.data.Dataset:
    """
    Using the provided split dataset, create a tf.data.Dataset.
    """
    features: NDArray = np.append(x_train, x_test, axis=0)
    labels: NDArray = np.append(y_train, y_test, axis=0)
    return tf.data.Dataset.from_tensor_slices((features, labels))


def load_cifar100() -> tf.data.Dataset:
    train, test = tf.keras.datasets.cifar100.load_data()
    return _dataset_from_split(train[0], train[1], test[0], test[1])


def load_cifar10() -> tf.data.Dataset:
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
    return features, labels


def shuffle_kaggle(kaggle: tf.data.Dataset) -> tf.data.Dataset:
    """
    Shuffles Kaggle Dataset and datasets derived from it via clustering.
    """
    kaggleSize = 197324
    return kaggle.shuffle(kaggleSize, seed=global_seed,
                          reshuffle_each_iteration=False)


def load_kaggle() -> tf.data.Dataset:
    """
    Create Kaggle as tf.data.Dataset from Numpy arrays
    """
    features, labels = _read_kaggle_data()
    return tf.data.Dataset.from_tensor_slices((features, labels))


def load_clustered_kaggle(numberOfClusters: int):
    """
    Load the Kaggle data and cluster it.
    """
    print(f"Clustering Kaggle with {numberOfClusters} classes..")
    kmeans = sklearn.cluster.MiniBatchKMeans(
        n_clusters=numberOfClusters,
        random_state=global_seed)
    features, _ = _read_kaggle_data()
    kaggleSize = 197324
    labels: NDArray = kmeans.fit_predict(features).reshape(kaggleSize, 1)
    return tf.data.Dataset.from_tensor_slices((features, labels))


def load_dataset(datasetName: str) -> tf.data.Dataset:
    """
    Load a dataset.

    Valid `datasetName` values are: "cifar10", "cifar100", "kaggle", "kaggle_2",
    "kaggle_10","kaggle_20","kaggle_50","kaggle_100".
    """
    datasetDir: str = join(dataDir, datasetName, "dataset")
    if isdir(datasetDir):
        print(f"Loading {datasetName} from disk.")
        return tf.data.experimental.load(datasetDir)

    print(f"Loading {datasetName}..")

    match datasetName:
        case "cifar10":
            dataset = load_cifar10()
        case "cifar100":
            dataset = load_cifar100()
        case "kaggle":
            dataset = load_kaggle()
        case "kaggle_2":
            dataset = load_clustered_kaggle(2)
        case "kaggle_10":
            dataset = load_clustered_kaggle(10)
        case "kaggle_20":
            dataset = load_clustered_kaggle(20)
        case "kaggle_50":
            dataset = load_clustered_kaggle(50)
        case "kaggle_100":
            dataset = load_clustered_kaggle(100)
        case _:
            raise ValueError  # TODO

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
    #  k = load_dataset("kaggle_20")
    #  print(k.element_spec)
