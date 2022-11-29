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
