import argparse
from typing import Dict

import datasets
import download
from configuration import Configuration


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        description='Launch a membership inference attack pipeline')
    parser.add_argument(
        '--config',
        help='Relative path to config file.',
    )

    return vars(parser.parse_args())


def set_seeds(config: Configuration):
    datasets.set_seed(config.seed)


def parse_config() -> Configuration:
    options = parse_args()
    try:
        config = Configuration.from_rel_path(options["config"])
        print("Using provided configuration.")
    except BaseException:
        config = Configuration.from_name("example.yml")
        print("Using default configuration.")
    return config


if __name__ == "__main__":

    # Meta/Preparing
    config = parse_config()
    set_seeds(config)

    # Download datasets
    download.download_all_datasets()

    # Prepare datasets for training
    kaggle = datasets.load_dataset("kaggle")
    kaggle = datasets.shuffle_kaggle(kaggle)  # Randomly sample training set

    # Train victim model

    # Launch MIA attack
