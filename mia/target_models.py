"""
.. include:: ../docs/target_models.md
"""

from os import environ, mkdir

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import datasets as ds

from os.path import dirname, isdir, join
import datetime

from typing import Dict
from tensorflow import keras
from tensorflow.python.framework import random_seed
from tensorflow.data import Dataset  # pyright: ignore
from tensorflow.keras.activations import tanh  # pyright: ignore
from tensorflow.keras.layers import Conv2D, Dense, InputLayer  # pyright: ignore
from tensorflow.keras.layers import MaxPool2D, Softmax, Flatten  # pyright: ignore
from tensorflow.keras import Sequential  # pyright: ignore

global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed
    random_seed.set_seed(global_seed)


class CifarModel(Sequential):
    """
    On CIFAR datasets, we train a standard convolutional neural
    network (CNN) with two convolution and max pooling layers
    plus a fully connected layer of size 128 and a Sof tMax layer.
    ‘We use Tanh as the activation function. We set the learning
    rate to 0.001, the learning rate decay to le — 07, and the
    maximum epochs of training to 100.
    """

    def __init__(self) -> None:
        super().__init__()
        activation = tanh
        batchSize = 100
        self.add(InputLayer(input_shape=(32, 32, 3), batch_size=batchSize))
        self.add(Conv2D(32, 3, activation=activation))
        self.add(MaxPool2D(pool_size=(2, 2)))
        self.add(Conv2D(32, 3, activation=activation))
        self.add(MaxPool2D(pool_size=(2, 2)))
        self.add(Flatten())
        self.add(Dense(128, activation=activation))
        self.add(Dense(10, activation=activation))
        self.add(Softmax())


class KaggleModel(Sequential):
    """
    On the purchase dataset (see Section VI-A), we train a fully
    connected neural network with one hidden layer of size 128
    and a SoftMax layer. We use Tanh as the activation function.
    ‘We set the learning rate to 0.001, the learning rate decay to
    1e — 07, and the maximum epochs of training to 200.
    """

    def __init__(self, output_size: int) -> None:
        super().__init__()
        activation = tanh
        batchSize = 100  # TODO: hardcoded
        self.add(InputLayer(input_shape=(600), batch_size=batchSize))
        self.add(Dense(128, activation=activation))
        self.add(Dense(output_size, activation=activation))
        self.add(Softmax())


def load_model(name: str, verbose=True) -> Sequential:
    """
    Load model from disk.

    The file name will be constructed from the `name` argument.
    """
    if verbose:
        print(f"Loading model {name} from disk.")
    filePath: str = join(dirname(__file__), "../models/target", name)
    return keras.models.load_model(filePath)


def save_model(name: str, model: Sequential) -> None:
    """
    Save model to disk.

    The file name will be constructed from the `name` argument.
    """
    folderPath: str = join(dirname(__file__),"../model/target")
    if not isdir(folderPath):
        mkdir(folderPath)
    filePath: str = join(folderPath, name)
    model.save(filePath)


def train_model(model: Sequential, modelName: str, trainData: Dataset,
                testData: Dataset, hyperpar: Dict):
    epochs: int = int(hyperpar["epochs"])
    learningRate: float = float(hyperpar["learningRate"])
    batchSize: int = int(hyperpar["batchSize"])
    print(
        f"Training model {modelName} for {epochs} epochs with learning rate {learningRate} and batch size {batchSize}.")

    optimizer = keras.optimizers.Adam(learning_rate=learningRate, name="Adam")
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.CategoricalAccuracy()]

    model.compile(optimizer, loss, metrics)
    trainData = trainData.batch(batchSize, drop_remainder=True)
    testData = testData.batch(batchSize, drop_remainder=True)
    log_dir = "logs/target/" + modelName
    cb = keras.callbacks.TensorBoard(histogram_freq=1, log_dir=log_dir)
    return model.fit(trainData, epochs=epochs, callbacks=[cb], validation_data=testData)


def evaluate_model(model: Sequential, dataset: Dataset):
    # TODO: batchSize is hardcoded
    batchSize = 100
    dataset = dataset.batch(batchSize, drop_remainder=True)
    return model.evaluate(dataset)


def get_model_name(config: Dict) -> str:
    modelConfig = config["targetModel"]["hyperparameters"]
    return \
        f'{config["targetDataset"]["name"]}_' + \
        f'classes_{config["targetModel"]["classes"]}_' + \
        f'lr_{modelConfig["learningRate"]}_' + \
        f'bs_{modelConfig["batchSize"]}_' + \
        f'epochs_{modelConfig["epochs"]}_' + \
        f'trainsize_{config["targetDataset"]["trainSize"]}'


def get_target_model(config: Dict, targetDataset) -> Sequential:
    """
    Try to load target model. If it doesn't work, train it.
    """
    try:
        print(f"Loading target model from disk.")
        modelName = get_model_name(config)
        model: KaggleModel = load_model(modelName, verbose=config["verbose"])

    except BaseException:
        print("Didn't work, retraining target model.")
        model: KaggleModel = train_target_model(config, targetDataset)

    return model


def train_target_model(config: Dict, targetDataset) -> Sequential:
    # TODO: Need to randomly select the 10.000 records used for training
    # TODO: Need to randomly select the 10.000 records used for testing

    targetDataset = ds.shuffle(targetDataset)
    dataConfig = config["targetDataset"]
    modelConfig = config["targetModel"]["hyperparameters"]

    modelName = get_model_name(config)
    trainDataName = modelName + "_train_data"
    testDataName = modelName + "_test_data"
    restDataName = modelName + "_rest_data"

    trainData = targetDataset.take(dataConfig["trainSize"])
    testData = targetDataset.skip(dataConfig["trainSize"]).take(dataConfig["testSize"])
    restData = targetDataset.skip(dataConfig["trainSize"]).skip(dataConfig["testSize"])

    ds.save_target(trainData,trainDataName)
    ds.save_target(trainData,testDataName)
    ds.save_target(restData,restDataName)

    if dataConfig["shuffle"]:
        trainData = ds.shuffle(trainData)

    model = KaggleModel(config["targetModel"]["classes"])

    train_model(model, modelName, trainData, testData, modelConfig)

    print("Saving target model to disk.")
    save_model(modelName, model)
    evaluate_model(model, testData)
    return model


if __name__ == "__main__":
    import argparse
    import configuration as con
    import datasets as ds

    parser = argparse.ArgumentParser(description='Train the target model.')
    parser.add_argument('--config', help='Relative path to config file.',)
    config = con.from_cli_options(vars(parser.parse_args()))
    set_seed(config["seed"])

    targetDataset = ds.load_dataset(config["targetDataset"]["name"])
    targetModel = get_target_model(config, targetDataset)
