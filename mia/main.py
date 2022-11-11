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

    targetTrainData, targetTestData = dataset.take(
        trainSize), dataset.skip(trainSize).take(testSize)

    if config["targetDataset"]["shuffle"]:
        targetTrainData = datasets.shuffle_dataset(targetTrainData, trainSize)

    # Construct target model
    targetEpochs: int = config["targetModel"]["hyperparameters"]["epochs"]
    targetBatchSize: int = config["targetModel"]["hyperparameters"]["batchSize"]
    targetLearningRate: float = config["targetModel"]["hyperparameters"]["learningRate"]
    targetTrainSize: int = config["targetDataset"]["trainSize"]
    targetModelName: str = f"lr_{targetLearningRate}_bs_{targetBatchSize}_epochs_{targetEpochs}_trainsize_{targetTrainSize}"

    if config["actions"]["trainTarget"]:
        targetModel: target_models.KaggleModel = target_models.KaggleModel(
            config["targetModel"]["classes"])

        target_models.train_model(
            targetModel,
            targetModelName,
            targetTrainData,
            targetTestData,
            config["targetModel"]["hyperparameters"])
        target_models.save_model(targetModelName, targetModel)
    else:
        targetModelName: str = config["targetModel"]["name"]
        targetModel: target_models.KaggleModel = target_models.load_model(
            targetModelName)

    # Evaluate target model
    if config["actions"]["testTarget"]:
        result = target_models.evaluate_model(targetModel, targetTestData)

    label_range = range(0, 100)

    # Generate attack data
    attackDataName = config["attackDataset"]["name"]

    if config["actions"]["generateAttackData"]:
        for label in label_range:
            try:
                attackTrainData, attackTestData = attack_data.from_target_data(
                    targetTrainData, targetTestData, targetModel, label)
            except BaseException:
                continue
            # Save
            attackDataNameTest = attackDataName + "_test" + f"_label_{label}"
            attackDataNameTrain = attackDataName + "_train" + f"_label_{label}"
            datasets.save_attack(attackTrainData, attackDataNameTrain)
            datasets.save_attack(attackTestData, attackDataNameTest)

    # Set up attack model
    epochs: int = config["attackModel"]["hyperparameters"]["epochs"]
    batchSize: int = config["attackModel"]["hyperparameters"]["batchSize"]
    learningRate: float = config["attackModel"]["hyperparameters"]["learningRate"]

    accs = []
    if config["actions"]["trainAttack"]:
        for label in label_range:
            attackDataNameTest = attackDataName + \
                "_test" + f"_label_{label}"
            attackDataNameTrain = attackDataName + \
                "_train" + f"_label_{label}"

            try:
                attackTestData = datasets.load_attack(attackDataNameTest)
                attackTrainData = datasets.load_attack(attackDataNameTrain)
            except BaseException:
                print(f"Aborting for label {label}.")
                continue
            attackModelName: str = f"lr_{learningRate}_bs_{batchSize}_epochs_{epochs}_label{label}"
            attackModel = attack_model.KaggleAttackModel(
                config["targetModel"]["classes"])
            try:
                attack_model.train_model(
                    attackModel,
                    attackModelName,
                    attackTrainData,
                    attackTestData,
                    config["attackModel"]["hyperparameters"])
            except BaseException:
                print(f"Aborting for label {label}.")
                continue
            #  attack_model.save_model(attackModelName, attackModel)
            #  attack_model.evaluate_model(attackModel, attackTestData)
    else:
        attackModelName: str = config["attackModel"]["name"]
        attackModel = attack_model.load_model(attackModelName)
