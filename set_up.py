import argparse
import glob
import os
import pathlib
import tarfile
from typing import Dict

import requests

import mia.datasets as datasets

topLevelDir = os.path.dirname(__file__)
miaDir = os.path.join(topLevelDir, "mia")
dataDir = os.path.join(topLevelDir, "data")


def generate_docs():
    docDir = "docs/build"
    print(f"Generating documentation into {docDir}.")
    import pdoc

    docsDirPath = pathlib.Path(os.path.join(topLevelDir, "docs/build"))
    pyFiles = glob.glob("*.py", root_dir=miaDir)

    curDir = os.path.curdir
    os.chdir(miaDir)
    pdoc.pdoc(*pyFiles, output_directory=docsDirPath)
    os.chdir(curDir)


def check_if_downloaded(datasetName: str) -> bool:
    files = datasets.DatasetFiles(datasetName)
    if path.isfile(files.numpyFeatures) and path.isfile(files.numpyLabels):
        print("Skipping, already downloaded.")
        return True
    return False


def download_raw_kaggle_data():
    kaggleDataDir = os.path.join(dataDir, "kaggle")
    kaggleCompressed = os.path.join(kaggleDataDir, "raw_data.tgz")
    kaggleRaw = os.path.join(kaggleDataDir, "raw_data")
    if not os.path.isfile(kaggleCompressed):
        url = 'https://github.com/OliverRoss/replicating_mia_datasets/raw/master/dataset_purchase.tgz'
        response = requests.get(url)
        with open(kaggleCompressed, mode='wb') as file:
            file.write(response.content)
    else:
        print("Skipping download, using local archive file.")

    if not os.path.isfile(kaggleRaw):
        tarfile.open(kaggleCompressed).extractall(kaggleDataDir)
        # "dataset_purchase" is the file name, we use the one in kaggleRaw
        os.rename(os.path.join(kaggleDataDir, "dataset_purchase"), kaggleRaw)
    else:
        print("Skipping extraction, using local raw data file.")


def download_kaggle():
    print("Downloading Kaggle Dataset.")
    if check_if_downloaded("kaggle"):
        return

    download_raw_kaggle_data()
    datasets.KagglePurchaseDataset()


def download_cifar10():
    print("Downloading CIFAR-10 Dataset.")
    if check_if_downloaded("cifar10"):
        return
    # Loading happens inside __init__
    datasets.Cifar10Dataset()


def download_cifar100():
    print("Downloading CIFAR-100 Dataset.")
    if check_if_downloaded("cifar100"):
        return
    # Loading happens inside __init__
    datasets.Cifar100Dataset()


def run_tests():
    print("Running tests from inside `mia/`.")
    import pytest
    curDir = os.path.curdir
    os.chdir(miaDir)
    returnCode = pytest.main(["-v", "-W", "ignore::DeprecationWarning"])
    os.chdir(curDir)
    if returnCode != pytest.ExitCode.OK:
        raise AssertionError("Tests did not succesfully run through.")


def parse_args() -> Dict:
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
        '--download',
        action='store_true',
        help='Download all datasets.'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests.'
    )

    return vars(parser.parse_args())


def make_data_dirs():
    if not os.path.isdir(dataDir):
        os.mkdir(dataDir)
    for dataset in ["cifar10", "cifar100", "kaggle"]:
        datasetDir = os.path.join(dataDir, dataset)
        if not os.path.isdir(datasetDir):
            os.mkdir(datasetDir)


def perform_options(opts: Dict):
    make_data_dirs()

    # Check if no option was provided -> do all options
    if True not in opts.values():
        for opt in opts.keys():
            opts[opt] = True
        opts["download"] = False

    if opts["download"]:
        download_kaggle()
        download_cifar10()
        download_cifar100()
    else:
        if opts['kaggle']:
            download_kaggle()
        if opts['cifar10']:
            download_cifar10()
        if opts['cifar100']:
            download_cifar100()
    if opts['doc']:
        generate_docs()
    if opts['test']:
        run_tests()


def main():
    options = parse_args()
    try:
        perform_options(options)
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Have you installed all dependencies? Use `python -m pip install -r requirements.txt`.") from e


if __name__ == "__main__":
    main()
