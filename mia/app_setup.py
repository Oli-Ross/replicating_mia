import target_models
import attack_model
import shadow_data
import attack_data
import datasets
from typing import Dict


def set_seeds(seed: int):
    datasets.set_seed(seed)
    target_models.set_seed(seed)
    attack_model.set_seed(seed)
    attack_data.set_seed(seed)
    shadow_data.set_seed(seed)


def _get_target_model_name(config: Dict) -> str:
    epochs: int = config["targetModel"]["hyperparameters"]["epochs"]
    batchSize: int = config["targetModel"]["hyperparameters"]["batchSize"]
    learningRate: float = config["targetModel"]["hyperparameters"]["learningRate"]
    trainSize: int = config["targetDataset"]["trainSize"]
    modelName: str = f"lr_{learningRate}_bs_{batchSize}_epochs_{epochs}_trainsize_{trainSize}"
    return modelName


def set_up_target_model(config: Dict, targetDataset):

    trainSize: int = config["targetDataset"]["trainSize"]
    testSize: int = config["targetDataset"]["testSize"]

    targetTrainData = targetDataset.take(trainSize)
    targetTestData = targetDataset.skip(trainSize).take(testSize)

    if config["targetDataset"]["shuffle"]:
        targetTrainData = datasets.shuffle(targetTrainData)

    if config["actions"]["trainTarget"]:
        targetModel: target_models.KaggleModel = target_models.KaggleModel(
            config["targetModel"]["classes"])
        targetModelName: str = _get_target_model_name(config)

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
        target_models.evaluate_model(targetModel, targetTestData)

    return targetModel


def _get_attack_model_name(label: int, config: Dict) -> str:
    epochs: int = config["attackModel"]["hyperparameters"]["epochs"]
    batchSize: int = config["attackModel"]["hyperparameters"]["batchSize"]
    learningRate: float = config["attackModel"]["hyperparameters"]["learningRate"]
    modelName: str = f"lr_{learningRate}_bs_{batchSize}_epochs_{epochs}_label{label}"
    return modelName


def load_attack_model(config: Dict):
    attackModelName: str = config["attackModel"]["name"]
    return attack_model.load_model(attackModelName)


def get_shadow_data(config: Dict, targetDataset,
                    targetModel) -> datasets.Dataset:
    shadowDataName: str = config["shadowDataset"]["name"]
    if config["actions"]["generateShadowData"]:
        method = config["shadowDataset"]["method"]
        if method == "noisy":
            shadowDataset: datasets.Dataset = shadow_data.generate_shadow_data_noisy(
                targetDataset, outputSize=500000)
        elif method == "hill_climbing":
            shadowDataset: datasets.Dataset = shadow_data.hill_climbing(
                targetModel,
                config["shadowDataset"]["size"],
                **config["shadowDataset"]["hill_climbing"]["hyperparameters"])
        else:
            raise ValueError(f"{method} is not a valid shadow data method.")
        datasets.save_shadow(shadowDataset, shadowDataName)
    else:
        shadowDataset: datasets.Dataset = datasets.load_shadow(shadowDataName)
    return shadowDataset
