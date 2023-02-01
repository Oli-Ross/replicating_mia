from os import environ
from os.path import isabs
import argparse
from typing import Dict, List, Tuple

from numpy._typing import NDArray
from tensorflow.keras.utils import to_categorical  # pyright: ignore
import tensorflow as tf
import numpy as np

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA


import download
import configuration as con
import target_models as tm
import shadow_data as sd
import datasets as ds
import attack_data as ad


def set_seeds(seed: int):
    ds.set_seed(seed)
    tm.set_seed(seed)
    sd.set_seed(seed)
    #  attack_model.set_seed(seed)
    #  attack_data.set_seed(seed)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(description='Launch a membership inference attack pipeline')
    parser.add_argument('--config', help='Relative path to config file.',)

    return vars(parser.parse_args())


def parse_config() -> Dict:
    options = parse_args()
    configFile = options["config"]
    try:
        if isabs(configFile):
            config = con.from_abs_path(configFile)
        else:
            config = con.from_rel_path(configFile)
        name = config["name"]
        print(f"Using configuration \"{name}\"")
    except BaseException:
        config = con.from_name("example.yml")
        print("Using default configuration.")
    return config


def get_target_model_name(config: Dict) -> str:
    modelConfig = config["targetModel"]["hyperparameters"]
    return \
        f'{config["targetDataset"]["name"]}_' + \
        f'lr_{modelConfig["learningRate"]}_' + \
        f'bs_{modelConfig["batchSize"]}_' + \
        f'epochs_{modelConfig["epochs"]}_' + \
        f'trainsize_{config["targetDataset"]["trainSize"]}'


def get_shadow_model_name(config: Dict, i: int):
    numModels: int = config["shadowModels"]["number"]
    split: float = config["shadowModels"]["split"]
    return "shadow_" + get_target_model_name(config) + f"_split_{split}_{i}_of_{numModels}"


def get_shadow_models_and_datasets(config: Dict, shadowDatasets: List[ds.Dataset]
                                   ) -> Tuple[List[tm.Sequential], List[Tuple[ds.Dataset, ds.Dataset]]]:
    """
    Tries to load shadow datasets from disk, alternatively trains from scratch.

    Returns 2 lists:
        models: the trained shadow models and a list of tuples, containing
        datasets: the training and test data for the corresponding shadow models

        E.g. models[0] is trained with datasets[0,0] and tested on datasets[0,1]
    """
    numModels: int = config["shadowModels"]["number"]
    split: float = config["shadowModels"]["split"]
    dataSize = shadowDatasets[0].cardinality().numpy()
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
            model: tm.KaggleModel = tm.load_model(modelName, verbose=False)
            trainData: ds.Dataset = ds.load_shadow(trainDataName, verbose=False)
            testData: ds.Dataset = ds.load_shadow(testDataName, verbose=False)

        except BaseException:
            print(f"Didn't work, training shadow model {i}.")
            dataset = shadowDatasets[i]
            trainData = dataset.take(trainSize)
            testData = dataset.skip(trainSize).take(testSize)

            # Shadow models have same architecture as target model
            model = tm.KaggleModel(config["targetModel"]["classes"])
            modelConfig = config["targetModel"]["hyperparameters"]

            tm.train_model(model, modelName, trainData, testData, modelConfig)

            print(f"Saving shadow model {i} and its data to disk.")
            tm.save_model(modelName, model)
            ds.save_shadow(trainData, trainDataName)
            ds.save_shadow(testData, testDataName)

            print(f"Evaluating shadow model {i}")
            tm.evaluate_model(model, testData)

        datasets.append((trainData, testData))
        models.append(model)

    return models, datasets


def get_target_model(config: Dict, targetDataset):
    dataConfig = config["targetDataset"]
    modelConfig = config["targetModel"]["hyperparameters"]
    modelName = get_target_model_name(config)

    try:
        print(f"Loading target model from disk.")
        model: tm.KaggleModel = tm.load_model(modelName, verbose=False)

    except BaseException:
        print("Didn't work, retraining target model.")

        trainData = targetDataset.take(dataConfig["trainSize"])
        testData = targetDataset.skip(dataConfig["trainSize"]).take(dataConfig["testSize"])

        if dataConfig["shuffle"]:
            trainData = ds.shuffle(trainData)

        model = tm.KaggleModel(config["targetModel"]["classes"])

        tm.train_model(model, modelName, trainData, testData, modelConfig)

        print("Saving target model to disk.")
        tm.save_model(modelName, model)
        tm.evaluate_model(model, testData)

    return model


def get_shadow_data(config: Dict, targetDataset, targetModel) -> ds.Dataset:
    shadowConfig = config["shadowDataset"]
    method = shadowConfig["method"]
    dataSize = shadowConfig["size"]
    hyperpars = shadowConfig[method]["hyperparameters"]

    if method == "noisy":
        dataName = f'{method}_fraction_{hyperpars["fraction"]}_size_{dataSize}'
        try:
            print("Loading shadow data from disk.")
            shadowData = ds.load_shadow(dataName, verbose=False)
        except BaseException:
            print("Loading failed, generating shadow data.")
            shadowData = sd.generate_shadow_data_noisy(targetDataset, dataSize, **hyperpars)
            ds.save_shadow(shadowData, dataName)
    elif method == "hill_climbing":
        dataName = \
            f'{method}_' + \
            f'kmax_{hyperpars["k_max"]}_' + \
            f'kmin_{hyperpars["k_min"]}_' + \
            f'confmin_{hyperpars["conf_min"]}_' + \
            f'rejmax_{hyperpars["rej_max"]}_' + \
            f'itermax_{hyperpars["iter_max"]}_' + \
            f'size_{dataSize}'
        try:
            shadowData = ds.load_shadow(dataName)
        except BaseException:
            print("Loading failed, generating shadow data.")
            shadowData = sd.hill_climbing(targetModel, dataSize, **hyperpars)
            ds.save_shadow(shadowData, dataName)
    else:
        raise ValueError(f"{method} is not a valid shadow data method.")

    return shadowData


def split_shadow_data(config: Dict, shadowData: ds.Dataset) -> List[ds.Dataset]:
    print("Splitting shadow data into subsets.")
    numSubsets = config["shadowModels"]["number"]
    return ds.split_dataset(shadowData, numSubsets)


def _get_attack_data_name(config: Dict, i):
    numModels: int = config["shadowModels"]["number"]
    numClasses = config["targetModel"]["classes"]
    split: float = config["shadowModels"]["split"]
    return get_target_model_name(config) + f"_split_{split}_with_{numModels}_models_{i}_of_{numClasses}"


def _save_attack_datasets(config: Dict, datasets: List[ds.Dataset]):
    numClasses = config["targetModel"]["classes"]
    assert numClasses == len(
        datasets), "List should contain 1 dataset per class"
    for index, dataset in enumerate(datasets):
        if index % 10 == 0:
            print(f"Saving attack dataset #{index}/{numClasses}")
        ds.save_attack(dataset, _get_attack_data_name(config, index))


def _load_attack_datasets(config: Dict):
    numClasses = config["targetModel"]["classes"]
    numDatasets = numClasses
    attackDatasets = []
    for i in range(numDatasets):
        attackDatasets.append(ds.load_attack(_get_attack_data_name(config, i), verbose=False))
    return attackDatasets


def predict_and_label_shadow_data(config: Dict,
                                  shadowModels: List[tm.Sequential],
                                  shadowDatasets: List[Tuple[ds.Dataset, ds.Dataset]]) -> List[ds.Dataset]:
    try:
        print("Loading attack data.")
        return _load_attack_datasets(config)
    except BaseException:
        print("Didn't work, reconstructing it.")
        attackDatasets = ad.from_shadow_models(config, shadowModels, shadowDatasets)
        _save_attack_datasets(config, attackDatasets)
        return attackDatasets


def _make_stats(attackDatasets: List[ds.Dataset]):
    inls = 0
    outls = 0
    for index, dataset in enumerate(attackDatasets):
        print(f"Set #{index}")
        outtmp = 0
        intmp = 0
        it = dataset.as_numpy_iterator()
        for x in it:
            if x[1][0] == 1:
                intmp += 1
            else:
                outtmp += 1
        print(f"    in:{intmp}, out:{outtmp}")
        inls += intmp
        outls += outtmp
    print(f"Total:")
    print(f"    in: {inls}, out: {outls}")


def main():

    config = parse_config()
    set_seeds(config["seed"])

    download.download_all_datasets()

    targetDataset = ds.load_dataset(config["targetDataset"]["name"])
    targetModel = get_target_model(config, targetDataset)
    shadowData = get_shadow_data(config, targetDataset, targetModel)
    shadowDatasets = split_shadow_data(config, shadowData)
    shadowModels, shadowDatasets = get_shadow_models_and_datasets(config, shadowDatasets)
    attackDatasets = predict_and_label_shadow_data(config, shadowModels, shadowDatasets)
    _make_stats(attackDatasets)
    breakpoint()


if __name__ == "__main__":
    main()
