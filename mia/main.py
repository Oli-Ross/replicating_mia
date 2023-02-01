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
import shadow_models as sm
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


def get_target_model(config: Dict, targetDataset):
    dataConfig = config["targetDataset"]
    modelConfig = config["targetModel"]["hyperparameters"]
    modelName = tm.get_model_name(config)

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
    shadowData = sd.get_shadow_data(config, targetDataset, targetModel)
    shadowDatasets = sd.split_shadow_data(config, shadowData)
    shadowModels, shadowDatasets = sm.get_shadow_models_and_datasets(config, shadowDatasets)
    attackDatasets = ad.get_attack_data(config, shadowModels, shadowDatasets)
    _make_stats(attackDatasets)
    breakpoint()


if __name__ == "__main__":
    main()
