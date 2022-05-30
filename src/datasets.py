import csv
from os import mkdir
from os.path import dirname, exists, join

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


class DatasetFiles:
    """
    Paths to files that hold the content of a datset.
    """

    def __init__(self, datasetName: str) -> None:
        """
        Construct the paths of the dataset files from its name.
        """

        currentDirectoryName = dirname(__file__)

        self.dataDirectory: str = join(
            currentDirectoryName,
            f"../data/{datasetName}")
        self.numpyFeatures: str = join(self.dataDirectory, "features.npy")
        self.numpyLabels: str = join(self.dataDirectory, "labels.npy")


class Dataset:
    """
    Base class for dataset representation.
    """
    size = 1
    dataDimensions = [1]
    labelDimension = 1
    datasetName = "default"

    def __init__(self) -> None:
        """
        Set up numpy arrays to hold the dataset, using the dataset format.
        """

        # This base class should not be instantiated, subclass it instead
        assert self.__class__ != Dataset

        self.files: DatasetFiles = DatasetFiles(self.datasetName)

        labelsArrayShape: list[int] = [self.labelDimension]
        labelsArrayShape.insert(0, self.size)
        featuresArrayShape: list[int] = self.dataDimensions.copy()
        featuresArrayShape.insert(0, self.size)

        self.labels: NDArray = np.zeros(labelsArrayShape)
        self.features: NDArray = np.zeros(featuresArrayShape)
        self.load()

        # self.features should not be flattened or its shape changed otherwise
        assert list(self.features.shape) == featuresArrayShape

    def load(self):
        """
        Load the dataset into the numpy arrays.
        """
        if exists(self.files.numpyFeatures) and exists(self.files.numpyLabels):
            self.load_numpy_from_file()
        else:
            self.load_external()
            self.save()

    def load_external(self):
        """
        Using an external source (e.g. file on disk or download), load the
        dataset.
        """
        raise NotImplementedError("Must be implemented by subclass.")

    def load_numpy_from_file(self):
        self.features: NDArray = np.load(self.files.numpyFeatures)
        self.labels: NDArray = np.load(self.files.numpyLabels)

    def save(self):
        """
        Save the arrays that hold the dataset to disk.
        """
        if not exists(self.files.dataDirectory):
            mkdir(self.files.dataDirectory)
        np.save(self.files.numpyFeatures, self.features)
        np.save(self.files.numpyLabels, self.labels)


class KagglePurchaseDataset(Dataset):
    """
    Kaggle's Acquire Valued Shoppers Challenge dataset of binary features.
    """

    datasetName: str = "purchase"
    size: int = 197324
    dataDimensions: list[int] = [600]
    numberOfLabels: int = 1

    def __init__(self) -> None:
        super().__init__()

    def load_external(self):
        self.load_raw_data_from_file()

    # TODO: Assumes specific CSV format
    def load_raw_data_from_file(self):
        rawData: str = join(self.files.dataDirectory, "raw_data")
        with open(rawData) as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                self.labels[0, index] = row[0]
                self.features[:, index] = row[1:]


class Cifar10Dataset(Dataset):
    """
    CIFAR-10 dataset of small RGB images.
    """

    datasetName: str = "cifar10"
    size: int = 60000
    dataDimensions: list[int] = [32, 32, 3]
    numberOfLabels: int = 1

    def __init__(self) -> None:
        super().__init__()

    def load_external(self):
        self.load_from_tensorflow()

    def load_from_tensorflow(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.features: NDArray = np.append(x_train, x_test, axis=0)
        self.labels: NDArray = np.append(y_train, y_test, axis=0)


class Cifar100Dataset(Dataset):
    """
    CIFAR-100 dataset of small RGB images.
    """

    datasetName: str = "cifar100"
    size: int = 60000
    dataDimensions: list[int] = [32, 32, 3]
    numberOfLabels: int = 1

    def __init__(self) -> None:
        super().__init__()

    def load_external(self):
        self.load_from_tensorflow()

    def load_from_tensorflow(self):
        # "Fine" label_mode for 100 classes as in MIA paper
        (x_train, y_train), (x_test, y_test) = \
            tf.keras.datasets.cifar100.load_data(label_mode='fine')

        self.features: NDArray = np.append(x_train, x_test, axis=0)
        self.labels: NDArray = np.append(y_train, y_test, axis=0)
