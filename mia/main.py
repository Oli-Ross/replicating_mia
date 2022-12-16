from os import environ
from os.path import isabs
import argparse
from typing import Dict, List, Tuple
from tensorflow.keras.utils import to_categorical  # pyright: ignore
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

    for i in range(numModels):
        modelName = "shadow_" + \
            get_target_model_name(config) + f"_split_{split}_{i}"
        trainDataName = modelName + "_train_data"
        testDataName = modelName + "_test_data"

        try:
            print(f"Trying to load shadow model \"{modelName}\" from disk.")
            model: tm.KaggleModel = tm.load_model(modelName)
            trainData: ds.Dataset = ds.load_shadow(trainDataName)
            testData: ds.Dataset = ds.load_shadow(testDataName)

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


def predict_and_label_shadow_data(config: Dict, shadowModels:
                                  List[tm.Sequential], shadowDatasets:
                                  List[Tuple[ds.Dataset, ds.Dataset]]) -> Tuple[ds.Dataset, ds.Dataset]:
    """
    Predicts the shadow data on the shadow models themselves and labels it with
    "in" and "out", for the attack model to train on.
    """
    numModels: int = config["shadowModels"]["number"]
    inData = []
    outData = []
    for i in range(numModels):
        model = shadowModels[i]
        trainData, testData = shadowDatasets[i]
        trainDataSize = trainData.cardinality().numpy()
        testDataSize = testData.cardinality().numpy()
        assert trainDataSize >= testDataSize, "Code assumes this."

        trainData = trainData.batch(trainDataSize)
        testData = testData.batch(testDataSize)
        trainPredictions = model.predict(trainData, batch_size=trainDataSize)
        testPredictions = model.predict(testData, batch_size=testDataSize)

        # Ensure arrays have same size
        trainPredictions = trainPredictions[:testDataSize]
        inData.append(trainPredictions)
        outData.append(testPredictions)

    # Merge into 1 array
    inData = np.array(inData).reshape((-1, 100))
    outData = np.array(outData).reshape((-1, 100))

    inLabels = to_categorical(np.repeat(1, inData.shape[0]), num_classes=2)
    outLabels = to_categorical(np.repeat(0, outData.shape[0]), num_classes=2)
    inDataset = ds.Dataset.from_tensor_slices((inData, inLabels))
    outDataset = ds.Dataset.from_tensor_slices((outData, outLabels))
    return inDataset, outDataset


def main():

    config = parse_config()
    set_seeds(config["seed"])

    download.download_all_datasets()

    targetDataset = ds.load_dataset(config["targetDataset"]["name"])
    targetModel = get_target_model(config, targetDataset)
    shadowData = get_shadow_data(config, targetDataset, targetModel)
    shadowDatasets = split_shadow_data(config, shadowData)
    shadowModels, shadowDatasets = get_shadow_models_and_datasets(
        config, shadowDatasets)
    inDataset, outDataset = predict_and_label_shadow_data(
        config, shadowModels, shadowDatasets)


if __name__ == "__main__":
    main()
