from os import environ
from typing import Dict

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import download
import target_models as tm
import datasets
import app_parse as parse
import app_setup as setup


def set_seeds(seed: int):
    datasets.set_seed(seed)
    tm.set_seed(seed)
    #  attack_model.set_seed(seed)
    #  attack_data.set_seed(seed)
    #  shadow_data.set_seed(seed)


def set_up_target_model(config: Dict, targetDataset):
    dataConfig = config["targetDataset"]
    modelConfig = config["targetModel"]["hyperparameters"]

    modelName = \
        f'{dataConfig["name"]}_' + \
        f'lr_{modelConfig["learningRate"]}_' + \
        f'bs_{modelConfig["batchSize"]}_' + \
        f'epochs_{modelConfig["epochs"]}_' + \
        f'trainsize_{dataConfig["trainSize"]}'

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
            trainData = datasets.shuffle(trainData)

        model = tm.KaggleModel(config["targetModel"]["classes"])

        tm.train_model(model, modelName, trainData, testData, modelConfig)
        tm.save_model(modelName, model)
        tm.evaluate_model(model, testData)

    return model


def main():

    config = parse.parse_config()
    set_seeds(config["seed"])

    download.download_all_datasets()

    targetDataset = datasets.load_dataset(config["targetDataset"]["name"])
    targetModel = set_up_target_model(config, targetDataset)
    shadowDataset = setup.get_shadow_data(config, targetDataset, targetModel)


if __name__ == "__main__":
    main()
