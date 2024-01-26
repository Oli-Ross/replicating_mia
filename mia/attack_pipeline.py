"""
.. include:: ../docs/attack_pipeline.md
"""

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

from typing import Dict, Tuple

import numpy as np
from tensorflow.data import Dataset  # pyright: ignore
from tensorflow.python.framework import random_seed

import datasets as ds
import target_models as tm
import attack_model as am

global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed
    random_seed.set_seed(global_seed)

def load_target_data(config:Dict) -> Tuple[Dataset, Dataset]:
    """
    Returns tuple trainData, restData.

    RestData is data unused for training and testing previously.
    """
    targetModelName = tm.get_model_name(config)
    targetTrainDataName = targetModelName + "_train_data"
    targetRestDataName = targetModelName + "_rest_data"
    targetTrainData = ds.load_target(targetTrainDataName)
    targetRestData = ds.load_target(targetRestDataName)
    return targetTrainData, targetRestData

def run_pipeline(attackModels, targetModel, targetTrainData, targetRestData):
    # (recall: % of records that were members were correctly inferred)
    # TODO: batchSize is hardcoded
    batchSizeTarget = 100
    batchSizeAttack = config["attackModel"]["hyperparameters"]["batchSize"]
    targetTrainDataSize = config["targetDataset"]["trainSize"]

    membersDataset = targetTrainData
    nonmembersDataset = targetRestData.take(targetTrainDataSize)

    memberPredictions = targetModel.predict(membersDataset.batch(batchSizeTarget))
    nonmemberPredictions = targetModel.predict(nonmembersDataset.batch(batchSizeTarget))

    memberAttackPredictions = []
    nonmemberAttackPredictions = []

    print("Predicting members.")
    for i, targetPrediction in enumerate(memberPredictions):
        label = np.argmax(targetPrediction)
        # select respective attack model, trained for that class
        attackModel = attackModels[label]
        modelInput = Dataset.from_tensors(targetPrediction).batch(batchSizeAttack)
        attackPrediction = attackModel.predict(modelInput,verbose = 0)
        memberAttackPredictions.append(np.argmax(attackPrediction))
        if i % 100 == 0:
            print(f"Predicted {i}/{targetTrainDataSize} records on attack model.")

    print("Predicting nonmembers.")
    for i, targetPrediction in enumerate(nonmemberPredictions):
        label = np.argmax(targetPrediction)
        # select respective attack model, trained for that class
        attackModel = attackModels[label]
        modelInput = Dataset.from_tensors(targetPrediction).batch(batchSizeAttack)
        attackPrediction = attackModel.predict(modelInput, verbose = 0)
        nonmemberAttackPredictions.append(np.argmax(attackPrediction))
        if i % 100 == 0:
            print(f"Predicted {i}/{targetTrainDataSize} records on attack model.")

    breakpoint()

    # precision: % of records inferred members, that are members

if __name__ == "__main__":
    import argparse
    import configuration as con

    parser = argparse.ArgumentParser(description='Run the attack pipeline on the target model.')
    parser.add_argument('--config', help='Relative path to config file.',)
    config = con.from_cli_options(vars(parser.parse_args()))
    set_seed(config["seed"])

    
    targetDataset = ds.load_dataset(config["targetDataset"]["name"])
    targetModel = tm.get_target_model(config, targetDataset)

    targetTrainData, targetRestData = load_target_data(config)

    attackModels = am.get_attack_models(config, [])

    run_pipeline(attackModels, targetModel, targetTrainData, targetRestData)
