"""
.. include:: ../docs/download.md
"""

import tarfile
from os import makedirs, path, rename

import requests
import tensorflow as tf

kaggleDataDir = path.join(path.dirname(__file__), "../data/kaggle")


def _extract_kaggle_data():
    kaggleCompressed = path.join(kaggleDataDir, "raw_data.tgz")
    if not path.isfile(kaggleCompressed):
        url = 'https://github.com/OliverRoss/replicating_mia_datasets/raw/master/dataset_purchase.tgz'
        response = requests.get(url)
        with open(kaggleCompressed, mode='wb') as file:
            file.write(response.content)


def _download_raw_kaggle_data():
    if not path.isdir(kaggleDataDir):
        makedirs(kaggleDataDir)
    kaggleRaw = path.join(kaggleDataDir, "raw_data")
    kaggleCompressed = path.join(kaggleDataDir, "raw_data.tgz")

    if not path.isfile(kaggleRaw):
        tarfile.open(kaggleCompressed).extractall(kaggleDataDir)
        # "dataset_purchase" is the file name, we use the one in kaggleRaw
        rename(path.join(kaggleDataDir, "dataset_purchase"), kaggleRaw)


def download_kaggle():
    _download_raw_kaggle_data()
    _extract_kaggle_data()


def download_cifar10():
    (_, _), (_, _) = tf.keras.datasets.cifar10.load_data()


def download_cifar100():
    (_, _), (_, _) = tf.keras.datasets.cifar100.load_data()


def download_all_datasets():
    print("Downloading all datasets.")
    download_cifar100()
    download_cifar10()
    download_kaggle()


if __name__ == "__main__":
    download_all_datasets()
