from typing import Dict, List, Tuple

import datasets as ds
import target_models as tm
import shadow_data as sd

import numpy as np


def get_shadow_model_name(config: Dict, i: int):
    numModels: int = config["shadowModels"]["number"]
    split: float = config["shadowModels"]["split"]
    return "shadow_" + tm.get_model_name(config) + f"_split_{split}_{i}_of_{numModels}"


def get_shadow_models_and_datasets(config: Dict, shadowDatasets: List[ds.Dataset]
                                   ) -> Tuple[List[tm.Sequential], List[Tuple[ds.Dataset, ds.Dataset]]]:
    """
    Tries to load shadow datasets from disk, alternatively trains from scratch.

    Returns 2 lists:
        models: the trained shadow models and a list of tuples, containing
        datasets: the training and test data for the corresponding shadow models

        E.g. models[0] is trained with datasets[0,0] and tested on datasets[0,1]
    """
    verbose = config["verbose"]
    numModels: int = config["shadowModels"]["number"]
    split: float = config["shadowModels"]["split"]
    dataSize = shadowDatasets[0].cardinality().numpy()
    assert dataSize != 0, "Loaded shadow dataset that seems empty."
    trainSize = np.ceil(split * dataSize)
    testSize = dataSize - trainSize
    datasets = []
    models = []

    print(f"Loading shadow models from disk.")
    for i in range(numModels):
        modelName = get_shadow_model_name(config, i)
        trainDataName = modelName + "_train_data"
        testDataName = modelName + "_test_data"

        try:
            model: tm.KaggleModel = tm.load_model(modelName, verbose=verbose)
            trainData: ds.Dataset = ds.load_shadow(trainDataName, verbose=verbose)
            testData: ds.Dataset = ds.load_shadow(testDataName, verbose=verbose)

        except BaseException:
            print(f"Didn't work, training shadow model {i}.")
            dataset = shadowDatasets[i]
            trainData = dataset.take(trainSize)
            testData = dataset.skip(trainSize).take(testSize)

            # Shadow models have same architecture as target model
            model = tm.KaggleModel(config["targetModel"]["classes"])
            modelConfig = config["targetModel"]["hyperparameters"]

            tm.train_model(model, modelName, trainData, testData, modelConfig)

            if config["cache_to_disk"]:
                print(f"Saving shadow model {i} and its data to disk.")
                tm.save_model(modelName, model)
                ds.save_shadow(trainData, trainDataName)
                ds.save_shadow(testData, testDataName)

            print(f"Evaluating shadow model {i}")
            tm.evaluate_model(model, testData)

        datasets.append((trainData, testData))
        models.append(model)

    return models, datasets
