import csv
from os import mkdir
from os.path import dirname, exists, join

import numpy as np
import tensorflow as tf
from numpy.typing import NDArray


class DatasetFormat:

    def __init__(self, size: int,
                 dataDimensions: list[int], numberOfLabels: int) -> None:
        self.size = size
        self.dataDimensions = dataDimensions
        self.numberOfLabels = numberOfLabels


class DatasetFiles:

    def __init__(self, datasetName: str) -> None:

        currentDirectoryName = dirname(__file__)

        self.dataDir = join(currentDirectoryName, f"../data/{datasetName}")
        self.rawData = join(self.dataDir, "raw_data")
        self.numpyFeatures = join(self.dataDir, "features.npy")
        self.numpyLabels = join(self.dataDir, "labels.npy")


class Dataset:

    def __init__(self, files: DatasetFiles, format: DatasetFormat) -> None:
        self.files = files
        self.format = format

        labelsArrayShape: list[int] = [format.numberOfLabels, format.size]
        featuresArrayShape: list[int] = format.dataDimensions.copy()
        featuresArrayShape.append(format.size)

        self.labels: NDArray = np.zeros(labelsArrayShape)
        self.features: NDArray = np.zeros(featuresArrayShape)

    def load(self):
        if exists(self.files.numpyFeatures) and exists(self.files.numpyLabels):
            self.load_numpy_from_file()
        else:
            self.load_external()
            self.save()

    def load_external(self):
        raise NotImplementedError("Must be implemented by subclass.")

    def save(self):
        if not exists(self.files.dataDir):
            mkdir(self.files.dataDir)
        self.save_numpy_to_file()

    def load_numpy_from_file(self):
        self.features = np.load(self.files.numpyFeatures)
        self.labels = np.load(self.files.numpyLabels)

    def save_numpy_to_file(self):
        np.save(self.files.numpyFeatures, self.features)
        np.save(self.files.numpyLabels, self.labels)


class KagglePurchaseDataset(Dataset):

    datasetName: str = "purchase"
    size: int = 197324
    dataDimensions: list[int] = [600]
    numberOfLabels: int = 1
    format: DatasetFormat = DatasetFormat(
        size, dataDimensions, numberOfLabels)
    files: DatasetFiles = DatasetFiles(datasetName)

    def __init__(self) -> None:
        super().__init__(self.files, self.format)
        self.load()

    def load_external(self):
        self.load_raw_data_from_file()

    # TODO: Assumes specific CSV format
    def load_raw_data_from_file(self):
        with open(self.files.rawData) as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                self.labels[0, index] = row[0]
                self.features[:, index] = row[1:]


class Cifar10Dataset(Dataset):

    datasetName: str = "cifar10"
    size: int = 60000
    dataDimensions: list[int] = [32, 32, 3]
    numberOfLabels: int = 1
    format: DatasetFormat = DatasetFormat(
        size, dataDimensions, numberOfLabels)
    files: DatasetFiles = DatasetFiles(datasetName)

    def __init__(self) -> None:
        super().__init__(self.files, self.format)
        self.load()

    def load_external(self):
        self.load_from_tensorflow()

    def load_from_tensorflow(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.features = np.append(x_train, x_test)
        self.labels = np.append(y_train, y_test)
