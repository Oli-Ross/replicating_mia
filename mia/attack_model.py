"""
.. include:: ../docs/attack_model.md
"""

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

from os.path import dirname, join

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras.activations import relu
from tensorflow.keras.layers import Dense, InputLayer, Softmax
from tensorflow.keras import Sequential

global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed


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
        self.add(InputLayer(input_shape=(numClasses)))
        self.add(Dense(64, activation=activation))
        self.add(Dense(2))
        self.add(Softmax())


def load_model(name: str) -> Sequential:
    """
    Load model from disk.

    The file name will be constructed from the `name` argument.
    """
    # TODO: make dir if it doesn't exist
    filePath: str = join(dirname(__file__), "../models/attack", name)
    return keras.models.load_model(filePath)


def save_model(name: str, model: Sequential) -> None:
    """
    Save model to disk.

    The file name will be constructed from the `name` argument.
    """
    # TODO: make dir if it doesn't exist
    filePath: str = join(dirname(__file__), "../models/attack", name)
    model.save(filePath)


def train_model(model: Sequential, dataset: Dataset, epochs=100,
                learningRate=0.01, batchSize=1):
    # TODO: Everything is hardcoded
    epochs = 10
    learningRate = 0.001
    batchSize = 1

    optimizer = keras.optimizers.Adam(learning_rate=learningRate, name="Adam")
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.CategoricalAccuracy()]

    model.compile(optimizer, loss, metrics)
    # TODO: drop_remainder: make sure dataset is still 50/50 in/out
    dataset = dataset.batch(batchSize, drop_remainder=True)
    return model.fit(dataset, epochs=epochs)


def evaluate_model(model: Sequential, dataset: Dataset):
    # TODO: batchSize is hardcoded
    batchSize = 1
    dataset = dataset.batch(batchSize, drop_remainder=False)
    return model.evaluate(dataset)
