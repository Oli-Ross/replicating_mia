import argparse
import glob
import os
import pathlib
from typing import Dict


def generate_docs():
    print("Generating documentation into /docs.")
    import pdoc

    topLevelDir = os.path.dirname(__file__)
    docsDirPath = pathlib.Path(os.path.join(topLevelDir, "docs"))
    pyFiles = glob.glob("src/*.py", root_dir=topLevelDir)

    pdoc.pdoc(*pyFiles, output_directory=docsDirPath)


def download_kaggle():
    print("Downloading Kaggle Dataset.")
    raise NotImplementedError()


def download_cifar10():
    print("Downloading CIFAR-10 Dataset.")
    import src.datasets as datasets
    datasets.Cifar10Dataset()


def download_cifar100():
    print("Downloading CIFAR-100 Dataset.")
    import src.datasets as datasets
    datasets.Cifar100Dataset()


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

    args = parser.parse_args()
    return args


def do_entire_setup():
    try:
        generate_docs()
    except(ModuleNotFoundError):
        print("pdocs does not seem to be available.")
        print("Skipping the generation of documentation.")
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


def main():
    options = parse_args()
    perform_options(vars(options))


if __name__ == "__main__":
    main()
