from os import environ
from os.path import isabs
import argparse
from typing import Dict, List
import numpy as np

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA


import download
import configuration as con
import target_models as tm
import shadow_data as sd
import datasets as ds


def set_seeds(seed: int):
    ds.set_seed(seed)
    tm.set_seed(seed)
    sd.set_seed(seed)
    #  attack_model.set_seed(seed)
    #  attack_data.set_seed(seed)


def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        description='Launch a membership inference attack pipeline')
    parser.add_argument(
        '--config',
        help='Relative path to config file.',
    )

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


def get_shadow_models(config: Dict, shadowDatasets:
                      List[ds.Dataset]) -> List[tm.Sequential]:
    numModels: int = config["shadowModels"]["number"]
    split: float = config["shadowModels"]["split"]
    dataSize = shadowDatasets[0].cardinality().numpy()
    trainSize = np.ceil(split * dataSize)
    testSize = dataSize - trainSize
    models = []

    for i in range(numModels):
        modelName = "shadow_" + \
            get_target_model_name(config) + f"_split_{split}_{i}"

        try:
            print(f"Trying to load shadow model \"{modelName}\" from disk.")
            model: tm.KaggleModel = tm.load_model(modelName)
        except BaseException:
            print(f"Didn't work, training shadow model {i}.")
            dataset = shadowDatasets[i]
            trainData = dataset.take(trainSize)
            testData = dataset.skip(trainSize).take(testSize)

            # Shadow models have same architecture as target model
            model = tm.KaggleModel(config["targetModel"]["classes"])
            modelConfig = config["targetModel"]["hyperparameters"]

            # TODO: this currently fails
            tm.train_model(model, modelName, trainData, testData, modelConfig)

            print(f"Saving shadow model {i} to disk.")
            tm.save_model(modelName, model)
            print(f"Evaluating shadow model {i}")
            tm.evaluate_model(model, testData)

        models.append(model)

    return models


def get_target_model(config: Dict, targetDataset):
    dataConfig = config["targetDataset"]
    modelConfig = config["targetModel"]["hyperparameters"]
    modelName = get_target_model_name(config)

    try:
        print(f"Trying to load model \"{modelName}\" from disk.")
        model: tm.KaggleModel = tm.load_model(modelName)

    except BaseException:
        print("Didn't work, retraining target model.")

        trainData = targetDataset.take(dataConfig["trainSize"])
        testData = targetDataset.skip(
            dataConfig["trainSize"]).take(
            dataConfig["testSize"])

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
            shadowData = ds.load_shadow(dataName)
        except BaseException:
            print("Loading failed, generating shadow data.")
            shadowData = sd.generate_shadow_data_noisy(
                targetDataset, dataSize, **hyperpars)
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


def split_shadow_data(
        config: Dict, shadowData: ds.Dataset) -> List[ds.Dataset]:
    print("Splitting shadow data into subsets.")
    numSubsets = config["shadowModels"]["number"]
    return ds.split_dataset(shadowData, numSubsets)


def main():

    config = parse_config()
    set_seeds(config["seed"])

    download.download_all_datasets()

    targetDataset = ds.load_dataset(config["targetDataset"]["name"])
    targetModel = get_target_model(config, targetDataset)
    shadowData = get_shadow_data(config, targetDataset, targetModel)
    shadowDatasets = split_shadow_data(config, shadowData)
    shadowModels = get_shadow_models(config, shadowDatasets)


if __name__ == "__main__":
    main()
