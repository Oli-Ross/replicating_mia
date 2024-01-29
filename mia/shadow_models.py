from typing import Dict, List, Tuple

import datasets as ds
import target_models as tm
import shadow_data as sd

from tensorflow.python.framework import random_seed

import numpy as np

global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed
    random_seed.set_seed(global_seed)


def get_shadow_model_name(config: Dict, i: int):
    numModels: int = config["shadowModels"]["number"]
    method: str = config["shadowDataset"]["method"]
    split: float = config["shadowModels"]["split"]
    return "shadow_" + method + tm.get_model_name(config) + f"_split_{split}_{i+1}_of_{numModels}"


def load_shadow_models_and_datasets(config: Dict) -> Tuple[List[tm.Sequential], List[Tuple[ds.Dataset, ds.Dataset]]]:
    verbose = config["verbose"]
    numModels: int = config["shadowModels"]["number"]
    datasets = []
    models = []

    print(f"Loading shadow models from disk.")
    for i in range(numModels):
        modelName = get_shadow_model_name(config, i)
        model: tm.KaggleModel = tm.load_model(modelName, verbose=verbose)

        trainDataName = modelName + "_train_data"
        testDataName = modelName + "_test_data"
        trainData: ds.Dataset = ds.load_shadow(trainDataName, verbose=verbose)
        testData: ds.Dataset = ds.load_shadow(testDataName, verbose=verbose)

        datasets.append((trainData, testData))
        models.append(model)

    return models, datasets


def train_shadow_models(config: Dict, shadowDatasets: List[ds.Dataset]
                        ) -> Tuple[List[tm.Sequential], List[Tuple[ds.Dataset, ds.Dataset]]]:

    numModels: int = config["shadowModels"]["number"]
    split: float = config["shadowModels"]["split"]
    dataSize = shadowDatasets[0].cardinality().numpy()
    assert dataSize != 0, "Loaded shadow dataset that seems empty."
    trainSize = np.ceil(split * dataSize)
    testSize = dataSize - trainSize
    datasets = []
    models = []

    for i in range(numModels):
        print(f"Training shadow model {i+1}.")

        modelName = get_shadow_model_name(config, i)
        trainDataName = modelName + "_train_data"
        testDataName = modelName + "_test_data"

        dataset = shadowDatasets[i]
        trainData = dataset.take(trainSize)
        testData = dataset.skip(trainSize).take(testSize)

        # Shadow models have same architecture as target model
        model = tm.KaggleModel(config["targetModel"]["classes"])
        modelConfig = config["targetModel"]["hyperparameters"]

        tm.train_model(model, modelName, trainData, testData, modelConfig)

        print(f"Saving shadow model {i+1} and its data to disk.")
        tm.save_model(modelName, model)
        ds.save_shadow(trainData, trainDataName)
        ds.save_shadow(testData, testDataName)

        print(f"Evaluating shadow model {i+1}")
        tm.evaluate_model(model, testData)

        datasets.append((trainData, testData))
        models.append(model)

    return models, datasets


def get_shadow_models_and_datasets(config: Dict, shadowDatasets: List[ds.Dataset]
                                   ) -> Tuple[List[tm.Sequential], List[Tuple[ds.Dataset, ds.Dataset]]]:
    """
    Tries to load shadow datasets from disk, alternatively trains from scratch.

    Returns 2 lists:
        models: the trained shadow models and a list of tuples, containing
        datasets: the training and test data for the corresponding shadow models

        E.g. models[0] is trained with datasets[0,0] and tested on datasets[0,1]
    """
    try:
        print("Trying to load shadow models and data from disk.")
        models, datasets = load_shadow_models_and_datasets(config)
    except BaseException:
        print("Didn't work, training shadow models.")
        models, datasets = train_shadow_models(config, shadowDatasets)

    return models, datasets


if __name__ == "__main__":
    import argparse
    import configuration as con
    import datasets as ds
    import target_models as tm
    import shadow_data as sd

    parser = argparse.ArgumentParser(description='Save one shadow dataset per model and train the models.')
    parser.add_argument('--config', help='Relative path to config file.',)
    config = con.from_cli_options(vars(parser.parse_args()))
    set_seed(config["seed"])

    shadowData = sd.load_shadow_data(config)
    shadowDatasets = sd.split_shadow_data(config, shadowData)
    shadowModels = get_shadow_models_and_datasets(config, shadowDatasets)
