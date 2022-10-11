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


def set_seeds(seed: int):
    datasets.set_seed(seed)
    target_models.set_seed(seed)
    shadow_data.set_seed(seed)


def parse_config() -> Dict:
    options = parse_args()
    try:
        config = con.from_rel_path(options["config"])
        name = config["name"]
        print(f"Using configuration \"{name}\"")
    except BaseException:
        config = con.from_name("example.yml")
        print("Using default configuration.")
    return config


if __name__ == "__main__":

    # Meta/Preparing
    config = parse_config()
    set_seeds(config["seed"])

    # Download datasets
    download.download_all_datasets()

    # Prepare dataset for training
    dataset = datasets.load_dataset(config["dataset"]["name"])
    trainSize: int = config["dataset"]["trainSize"]
    testSize: int = config["dataset"]["testSize"]
    train, test = dataset.take(trainSize), dataset.skip(
        trainSize).take(testSize)

    # Construct target model
    model_name: str = config["targetModel"]["name"]

    if config["actions"]["trainTarget"]:
        model: target_models.KaggleModel = target_models.KaggleModel(100)
        target_models.train_model(model, dataset,
                                  config["targetModel"]["hyperparameters"])
        target_models.save_model(model_name, model)
    else:
        model: target_models.KaggleModel = target_models.load_model(model_name)

    # Evaluate target model
    if config["actions"]["testTarget"]:
        result = target_models.evaluate_model(model, test)

    # Generate shadow data
    # (Skipped for now)
    #  my_shadow_data = shadow_data.generate_shadow_data_model(model, size=3)

    # Train shadow models
    # (Skipped for now)

    # Generate attack data

    # Set up attack model
    # Train attack model
    # Launch MIA attack
