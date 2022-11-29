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
