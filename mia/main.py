import argparse
from typing import Dict

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import datasets
import download
import target_models
import attack_model
import shadow_data
import attack_data
import configuration as con
import tensorflow as tf
from os.path import join


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
    attack_model.set_seed(seed)
    attack_data.set_seed(seed)
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

    # Load + split dataset for training
    dataset = datasets.load_dataset(config["targetDataset"]["name"])
    trainSize: int = config["targetDataset"]["trainSize"]
    testSize: int = config["targetDataset"]["testSize"]

    if config["targetDataset"]["shuffle"]:
        dataset = datasets.shuffle_dataset(dataset, trainSize)

    targetTrainData, targetTestData = dataset.take(
        trainSize), dataset.skip(trainSize).take(testSize)

    # Construct target model
    targetModelName: str = config["targetModel"]["name"]

    targetModelType: str = config["targetModel"]["type"]
    if config["actions"]["trainTarget"]:
        if targetModelType == "kaggle":
            targetModel: target_models.KaggleModel = target_models.KaggleModel(
                config["targetModel"]["classes"])
        elif targetModelType == "cifar":
            targetModel: target_models.KaggleModel = target_models.CifarModel()
        else:
            raise ValueError(f"{targetModelType} not known model type.")

        target_models.train_model(
            targetModel,
            targetTrainData,
            targetTestData,
            config["targetModel"]["hyperparameters"])
        target_models.save_model(targetModelName, targetModel)
    else:
        targetModel: target_models.KaggleModel = target_models.load_model(
            targetModelName)

    # Evaluate target model
    if config["actions"]["testTarget"]:
        result = target_models.evaluate_model(targetModel, targetTestData)

    # Generate shadow data
    # (Skipped for now)
    #  my_shadow_data = shadow_data.generate_shadow_data_model(model, size=3)

    # Train shadow models
    # (Skipped for now)

    # Generate attack data
    label: int = 18
    attackDataName = config["attackDataset"]["name"]
    attackDataNameTest = attackDataName + "_test"
    attackDataNameTrain = attackDataName + "_train"

    if config["actions"]["generateAttackData"]:
        attackTrainData, attackTestData = attack_data.from_target_data(
            targetTrainData, targetTestData, targetModel, label)
        # Save
        datasets.save_attack(attackTrainData, attackDataNameTrain)
        datasets.save_attack(attackTestData, attackDataNameTest)
    else:
        # Load
        attackTestData = datasets.load_attack(attackDataNameTest)
        attackTrainData = datasets.load_attack(attackDataNameTrain)

    # Set up attack model

    epochs: int = config["attackModel"]["hyperparameters"]["epochs"]
    batchSize: int = config["attackModel"]["hyperparameters"]["batchSize"]
    learningRate: float = config["attackModel"]["hyperparameters"]["learningRate"]
    attackModelName: str = f"lr_{learningRate}_bs_{batchSize}_epochs_{epochs}"

    if config["actions"]["trainAttack"]:
        if targetModelType == "kaggle":
            attackModel = attack_model.KaggleAttackModel(
                config["targetModel"]["classes"])
            attack_model.train_model(
                attackModel,
                attackModelName,
                attackTrainData,
                attackTestData,
                config["attackModel"]["hyperparameters"])
            attack_model.save_model(attackModelName, attackModel)
        else:
            raise NotImplementedError
    else:
        attackModelName: str = config["attackModel"]["name"]
        attackModel = attack_model.load_model(attackModelName)

    # Launch MIA attack
    if config["actions"]["testAttack"]:
        resultAttack = attack_model.evaluate_model(attackModel, attackTestData)
