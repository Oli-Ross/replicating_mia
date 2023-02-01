"""
.. include:: ../docs/attack_model.md
"""

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

from os.path import dirname, join
import datetime

from typing import Dict
from tensorflow import keras
from tensorflow.data import Dataset  # pyright: ignore
from tensorflow.python.framework import random_seed
from tensorflow.keras.activations import relu  # pyright: ignore
from tensorflow.keras.initializers import glorot_uniform  # pyright: ignore
from tensorflow.keras.layers import Dense, InputLayer, Softmax  # pyright: ignore
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


class KaggleAttackModel(Sequential):
    """
    Architecture:
        Fully connected NN,
        1 hiddenlayer, size 64,
        ReLU activation
        Softmax LAyer

    One model for each class
    """

    def __init__(self, numClasses: int) -> None:
        super().__init__()
        activation = relu
        initializer = glorot_uniform
        self.add(InputLayer(input_shape=(numClasses)))
        self.add(Dense(64, activation=activation,
                 kernel_initializer=initializer))
        self.add(Dense(2, kernel_initializer=initializer))
        self.add(Softmax())


def load_model(name: str) -> Sequential:
    """
    Load model from disk.

    The file name will be constructed from the `name` argument.
    """
    filePath: str = join(dirname(__file__), "../models/attack", name)
    return keras.models.load_model(filePath)


def save_model(name: str, model: Sequential) -> None:
    """
    Save model to disk.

    The file name will be constructed from the `name` argument.
    """
    filePath: str = join(dirname(__file__), "../models/attack", name)
    model.save(filePath)


def train_model(model: Sequential, modelName: str, trainData: Dataset,
                testData: Dataset, hyperpar: Dict):
    epochs: int = int(hyperpar["epochs"])
    learningRate: float = float(hyperpar["learningRate"])
    batchSize: int = int(hyperpar["batchSize"])

    optimizer = keras.optimizers.Adam(name="Adam", learning_rate=learningRate)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = ["accuracy"]

    model.compile(optimizer, loss, metrics)
    # TODO: drop_remainder: make sure dataset is still 50/50 in/out
    trainData = trainData.batch(batchSize, drop_remainder=True)
    testData = testData.batch(batchSize, drop_remainder=True)
    log_dir = "logs/attack/" + modelName
    cb = keras.callbacks.TensorBoard(histogram_freq=1, log_dir=log_dir)
    return model.fit(trainData, epochs=epochs, callbacks=[cb], validation_data=testData)


def evaluate_model(model: Sequential, dataset: Dataset):
    # TODO: batchSize is hardcoded
    batchSize = 10
    dataset = dataset.batch(batchSize, drop_remainder=False)
    return model.evaluate(dataset)
