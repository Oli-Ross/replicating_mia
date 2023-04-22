"""
.. include:: ../docs/download.md
"""

import tarfile
from os import makedirs, path, rename
from os.path import join, dirname, isdir

import requests
import tensorflow as tf

dataDir = join(dirname(__file__), "../data")
kaggleDir = join(dataDir, "kaggle")
cifar10DataDir: str = join(dataDir, "cifar10", "dataset")
cifar100DataDir: str = join(dataDir, "cifar100", "dataset")


def _download_raw_kaggle_data():
    kaggleCompressed = path.join(kaggleDir, "raw_data.tgz")
    if not path.isfile(kaggleCompressed):
        url = 'https://github.com/OliverRoss/replicating_mia_datasets/raw/master/dataset_purchase.tgz'
        response = requests.get(url)
        with open(kaggleCompressed, mode='wb') as file:
            file.write(response.content)


def _extract_kaggle_data():
    kaggleRaw = path.join(kaggleDir, "raw_data")
    kaggleCompressed = path.join(kaggleDir, "raw_data.tgz")

    if not path.isfile(kaggleRaw):
        tarfile.open(kaggleCompressed).extractall(kaggleDir)
        # "dataset_purchase" is the file name, we use the one in kaggleRaw
        rename(path.join(kaggleDir, "dataset_purchase"), kaggleRaw)


def download_kaggle():
    if not path.isdir(kaggleDir):
        makedirs(kaggleDir)
    _download_raw_kaggle_data()
    _extract_kaggle_data()


def download_cifar10():
    if not isdir(cifar10DataDir):
        (_, _), (_, _) = tf.keras.datasets.cifar10.load_data()


def download_cifar100():
    if not isdir(cifar100DataDir):
        (_, _), (_, _) = tf.keras.datasets.cifar100.load_data()


def download_all_datasets():
    print("Downloading all datasets.")
    download_cifar100()
    download_cifar10()
    download_kaggle()


def download_dataset(datasetName: str):
    if datasetName == "cifar10":
        download_cifar10()
    elif datasetName == "cifar100":
        download_cifar100()
    elif "kaggle" in datasetName:
        download_kaggle()
    else:
        raise ValueError(f"{datasetName} is not a known dataset.")


if __name__ == "__main__":
    import argparse
    import configuration as con
    parser = argparse.ArgumentParser(description='Make sure the needed dataset is downloaded.')
    parser.add_argument('--config', help='Relative path to config file.',)
    config = con.from_cli_options(vars(parser.parse_args()))
    dataName = config["targetDataset"]["name"]
    print(f"Downloading data for {dataName}, if necessary.")
    download_dataset(dataName)
