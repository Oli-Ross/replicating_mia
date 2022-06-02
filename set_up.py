from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import argparse
import glob
import os
import pathlib
import tarfile
from typing import Dict, Tuple

import requests

topLevelDir = os.path.dirname(__file__)


def generate_docs():
    print("Generating documentation into /docs.")
    import pdoc

    docsDirPath = pathlib.Path(os.path.join(topLevelDir, "docs"))
    pyFiles = glob.glob("mia/*.py", root_dir=topLevelDir)

    pdoc.pdoc(*pyFiles, output_directory=docsDirPath)


def download_kaggle_as_tgz(destinationFilePath: str):
    url = 'https://github.com/OliverRoss/replicating_mia_datasets/raw/master/dataset_purchase.tgz'
    response = requests.get(url)
    with open(destinationFilePath, mode='wb') as kaggleFile:
        kaggleFile.write(response.content)


def extract_tgz_to_dir(tarFileName: str, destDir: str):
    os.chdir(destDir)
    compressedFile = tarfile.open(tarFileName)
    compressedFile.extractall()
    os.rename("dataset_purchase", "raw_data")


def set_up_kaggle_directory() -> Tuple[str, str]:
    dataDir = os.path.join(topLevelDir, "data", "purchase")
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
    kaggleFileName = os.path.join(dataDir, "raw_data.tgz")
    return kaggleFileName, dataDir


def download_kaggle():
    print("Downloading Kaggle Dataset.")

    kaggleFileName, dataDir = set_up_kaggle_directory()
    download_kaggle_as_tgz(destinationFilePath=kaggleFileName)
    extract_tgz_to_dir(kaggleFileName, dataDir)


def download_cifar10():
    print("Downloading CIFAR-10 Dataset.")
    import tensorflow as tf
    (_, _), (_, _) = tf.keras.datasets.cifar10.load_data()


def download_cifar100():
    print("Downloading CIFAR-100 Dataset.")
    import tensorflow as tf
    (_, _), (_, _) = tf.keras.datasets.cifar100.load_data(label_mode='fine')


def run_tests():
    import pytest
    pytest.main(["-v", "-W", "ignore::DeprecationWarning"])


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare repository for MIA attack. Without provided option, all actions are performed.')
    parser.add_argument(
        '--doc',
        action='store_true',
        help='Generate the documentation.',
    )
    parser.add_argument(
        '--kaggle',
        action='store_true',
        help='Download Kaggle dataset.'
    )
    parser.add_argument(
        '--cifar10',
        action='store_true',
        help='Download CIFAR-10 dataset.'
    )
    parser.add_argument(
        '--cifar100',
        action='store_true',
        help='Download CIFAR-100 dataset.'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests.'
    )

    args = parser.parse_args()
    return args


def do_entire_setup():
    try:
        generate_docs()
    except(ModuleNotFoundError):
        print("pdocs does not seem to be available.")
        print("Skipping the generation of documentation.")
    run_tests()
    download_cifar10()
    download_cifar100()
    download_kaggle()


def perform_options(opts: Dict):

    noOptionsProvided = True
    for opt in opts:
        if opts[opt]:
            noOptionsProvided = False

    if noOptionsProvided:
        do_entire_setup()
    else:
        if opts['doc']:
            generate_docs()
        if opts['kaggle']:
            download_kaggle()
        if opts['cifar10']:
            download_cifar10()
        if opts['cifar100']:
            download_cifar100()
        if opts['test']:
            run_tests()


def main():
    options = parse_args()
    try:
        perform_options(vars(options))
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Have you installed all dependencies? Use `python -m pip install -r requirements.txt`.") from e


if __name__ == "__main__":
    main()
