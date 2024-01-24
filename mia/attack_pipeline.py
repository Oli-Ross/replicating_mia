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
    # TODO: batchSize is hardcoded
    batchSize = 100
    numClasses = config["targetModel"]["classes"]
    verbose = config["verbose"]
    targetTrainDataSize = config["targetDataset"]["trainSize"]

    membersDataset = targetTrainData.batch(batchSize)
    nonmembersDataset = targetRestData.take(targetTrainDataSize).batch(batchSize)

    memberPredictions = targetModel.predict(membersDataset)
    nonmemberPredictions = targetModel.predict(nonmembersDataset)

    memberLabels = np.argmax(memberPredictions, axis = 1)
    nonmemberLabels = np.argmax(nonmemberPredictions, axis = 1)

    for i in range(numClasses):
        pass

if __name__ == "__main__":
    import argparse
    import configuration as con
    import attack_data as ad

    parser = argparse.ArgumentParser(description='Run the attack pipeline on the target model.')
    parser.add_argument('--config', help='Relative path to config file.',)
    config = con.from_cli_options(vars(parser.parse_args()))
    set_seed(config["seed"])

    
    targetDataset = ds.load_dataset(config["targetDataset"]["name"])
    targetModel = tm.get_target_model(config, targetDataset)

    targetTrainData, targetRestData = load_target_data(config)

    attackDatasets = ad.load_attack_data(config)
    attackModels = am.get_attack_models(config, attackDatasets)

    run_pipeline(attackModels, targetModel, targetTrainData, targetRestData)
