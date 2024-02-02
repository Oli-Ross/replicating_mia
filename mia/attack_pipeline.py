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

import utils
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

def run_pipeline(targetModel, targetTrainData, targetRestData):
    # TODO: batchSize is hardcoded
    numClasses = config["targetModel"]["classes"]
    batchSizeTarget = 100
    batchSizeAttack = config["attackModel"]["hyperparameters"]["batchSize"]
    targetTrainDataSize = config["targetDataset"]["trainSize"]

    hash = utils.hash(str(config))

    try:
        memberAttackPredictions = ds.load_numpy_array(f"{hash}_memberAttackPredictions.npy")
        nonmemberAttackPredictions = ds.load_numpy_array(f"{hash}_nonmemberAttackPredictions.npy")

    except:
        attackModels = am.get_attack_models(config, [])

        membersDataset = targetTrainData
        nonmembersDataset = targetRestData.take(targetTrainDataSize)

        memberTargetPredictions = targetModel.predict(membersDataset.batch(batchSizeTarget))
        nonmemberTargetPredictions = targetModel.predict(nonmembersDataset.batch(batchSizeTarget))

        memberAttackPredictions = [[] for _ in range(numClasses)]
        nonmemberAttackPredictions = [[] for _ in range(numClasses)]

        print("Predicting members.")
        for i, targetPrediction in enumerate(memberTargetPredictions):
            label = np.argmax(targetPrediction)
            # select respective attack model, trained for that class
            attackModel = attackModels[label]
            modelInput = Dataset.from_tensors(targetPrediction).batch(batchSizeAttack)
            attackPrediction = attackModel.predict(modelInput,verbose = 0)
            memberAttackPredictions[label].append(np.argmax(attackPrediction))
            if i % 100 == 0 and config["verbose"]:
                print(f"Predicted {i}/{targetTrainDataSize} member records on attack model.")

        print("Predicting nonmembers.")
        for i, targetPrediction in enumerate(nonmemberTargetPredictions):
            label = np.argmax(targetPrediction)
            # select respective attack model, trained for that class
            attackModel = attackModels[label]
            modelInput = Dataset.from_tensors(targetPrediction).batch(batchSizeAttack)
            attackPrediction = attackModel.predict(modelInput, verbose = 0)
            nonmemberAttackPredictions[label].append(np.argmax(attackPrediction))
            if i % 100 == 0 and config["verbose"]:
                print(f"Predicted {i}/{targetTrainDataSize} nonmember records on attack model.")

        ds.save_numpy_array("{hash}_memberAttackPredictions.npy",memberAttackPredictions)
        ds.save_numpy_array("{hash}_nonmemberAttackPredictions.npy",nonmemberAttackPredictions)

    precisionPerClass = [None for _ in range(numClasses)]
    recallPerClass = [None for _ in range(numClasses)]

    for _class in range(numClasses):
        memberAttackPrediction = memberAttackPredictions[_class]
        if memberAttackPrediction:
            recallPerClass[_class] = 1 - np.average(memberAttackPrediction)

        nonmemberAttackPrediction = nonmemberAttackPredictions[_class]
        if nonmemberAttackPrediction:
            membersInferredAsMembers = len(memberAttackPrediction) - np.count_nonzero(memberAttackPrediction)
            nonmembersInferredAsMembers = len(nonmemberAttackPrediction) - np.count_nonzero(nonmemberAttackPrediction)
            precisionPerClass[_class] = membersInferredAsMembers / (membersInferredAsMembers + nonmembersInferredAsMembers)

    membersInferredAsMembers = targetTrainDataSize - sum([sum(x) for x in memberAttackPredictions])
    nonmembersInferredAsMembers = targetTrainDataSize - sum([sum(x) for x in nonmemberAttackPredictions])
    totalRecall = membersInferredAsMembers/targetTrainDataSize
    totalPrecision = membersInferredAsMembers / (membersInferredAsMembers + nonmembersInferredAsMembers)
    return totalPrecision, totalRecall, precisionPerClass, recallPerClass

def process_results(precision, recall, precisionPerClass, recallPerClass):

    precisionPerClassWithoutNone = [x for x in precisionPerClass if x]
    recallPerClassWithoutNone = [x for x in recallPerClass if x]

    hash = utils.hash(str(config))
    with open(f"{hash}_recallPerClass.csv",'w') as file:
        file.write(f"Recall (Overall:{recall})\n")
        for recall in sorted(recallPerClassWithoutNone):
            file.write(f"{recall}\n")
    with open(f"{hash}_precisionPerClass.csv",'w') as file:
        file.write(f"Precision (Overall: {precision})\n")
        for precision in sorted(precisionPerClassWithoutNone):
            file.write(f"{precision}\n")

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

    precision, recall, precisionPerClass, recallPerClass = run_pipeline(targetModel, targetTrainData, targetRestData)

    process_results(precision, recall, precisionPerClass, recallPerClass)
