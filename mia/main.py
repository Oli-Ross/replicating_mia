import argparse
from typing import Dict

import datasets
import download
import target_models
import shadow_data
import configuration as con


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        description='Launch a membership inference attack pipeline')
    parser.add_argument(
        '--config',
        help='Relative path to config file.',
    )

    return vars(parser.parse_args())


def set_seeds(config: Dict):
    datasets.set_seed(config["seed"])
    target_models.set_seed(config["seed"])
    shadow_data.set_seed(config["seed"])


def parse_config() -> Dict:
    options = parse_args()
    try:
        config = con.from_rel_path(options["config"])
        print("Using provided configuration.")
    except BaseException:
        config = con.from_name("example.yml")
        print("Using default configuration.")
    return config


if __name__ == "__main__":

    # Meta/Preparing
    config = parse_config()
    set_seeds(config)

    # Download datasets
    download.download_all_datasets()

    # Prepare datasets for training
    cifar10 = datasets.load_dataset("cifar10")
    trainSize: int = 10000
    train, test = cifar10.take(trainSize), cifar10.skip(trainSize)

    # Train target model
    cifar10Model = target_models.CifarModel()
    target_models.train_model(cifar10Model, cifar10, epochs=1)

    # Launch MIA attack
