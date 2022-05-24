import csv
from os.path import exists

import numpy as np
from numpy.typing import NDArray


class DatasetFormat:
    def __init__(self, size: int, dataFormat: list[int], numberOfLabels: int) -> None:
        self.size = size
        self.dataFormat = dataFormat
        self.numberOfLabels = numberOfLabels


class DatasetFiles:
    def __init__(self, datasetName: str) -> None:
        self.rawData = f"../data/{datasetName}/raw_data"
        self.numpyFeatures = f"../data/{datasetName}/features.npy"
        self.numpyLabels = f"../data/{datasetName}/labels.npy"


class Dataset:

    def __init__(self, files: DatasetFiles, format: DatasetFormat) -> None:

        self.files = files
        self.format = format

        labelsArrayShape: list[int] = [format.numberOfLabels, format.size]
        featuresArrayShape: list[int] = format.dataFormat.copy()
        featuresArrayShape.append(format.size)

        self.labels: NDArray = np.zeros(labelsArrayShape)
        self.features: NDArray = np.zeros(featuresArrayShape)

    def load(self):
        if exists(self.files.numpyFeatures) and exists(self.files.numpyLabels):
            self.load_numpy_from_file()
        else:
            pass

    def load_numpy_from_file(self):
        self.features = np.load(self.files.numpyFeatures)
        self.labels = np.load(self.files.numpyLabels)

    def save_numpy_to_file(self):
        np.save(self.files.numpyFeatures, self.features)
        np.save(self.files.numpyLabels, self.labels)


class KagglePurchaseDataset(Dataset):

    datasetName: str = "purchase"
    size: int = 197324
    dataFormat: list[int] = [600]
    numberOfLabels: int = 1
    format: DatasetFormat = DatasetFormat(
        size, dataFormat, numberOfLabels)
    files: DatasetFiles = DatasetFiles(datasetName)

    def __init__(self) -> None:
        super().__init__(self.files, self.format)
        self.load()

    def load(self):
        if exists(self.files.numpyFeatures) and exists(self.files.numpyLabels):
            self.load_numpy_from_file()
        else:
            self.load_raw_data_from_file()

    # TODO: Assumes specific CSV format
    def load_raw_data_from_file(self):
        with open(self.files.rawData) as file:
            reader = csv.reader(file)
            for index, row in enumerate(reader):
                self.labels[index] = row[0]
                self.features[index] = row[1:]


class Cifar10Dataset(Dataset):

    datasetName: str = "purchase"
    size: int = 197324
    dataFormat: list[int] = [32, 32, 3]
    numberOfLabels: int = 1
    format: DatasetFormat = DatasetFormat(
        size, dataFormat, numberOfLabels)
    files: DatasetFiles = DatasetFiles(datasetName)

    def __init__(self) -> None:
        super().__init__(self.files, self.format)
        self.load()

    def load(self):
        if exists(self.files.numpyFeatures) and exists(self.files.numpyLabels):
            self.load_numpy_from_file()
        else:
            self.load_from_tensorflow()

    def load_from_tensorflow(self):
        pass
