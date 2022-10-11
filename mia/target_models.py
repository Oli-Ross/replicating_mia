"""
.. include:: ../docs/target_models.md
"""

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

from os.path import dirname, join

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Conv2D, Dense, InputLayer, MaxPool2D, Softmax, Flatten
from tensorflow.keras import Sequential

global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed


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
        batchSize = 100
        self.add(InputLayer(input_shape=(600), batch_size=batchSize))
        self.add(Dense(128, activation=activation))
        self.add(Dense(output_size, activation=activation))
        self.add(Softmax())


def load_model(name: str) -> Sequential:
    """
    Load model from disk.

    The file name will be constructed from the `name` argument.
    """
    # TODO: make dir if it doesn't exist
    filePath: str = join(dirname(__file__), "../models/target", name)
    return keras.models.load_model(filePath)


def save_model(name: str, model: Sequential) -> None:
    """
    Save model to disk.

    The file name will be constructed from the `name` argument.
    """
    # TODO: make dir if it doesn't exist
    filePath: str = join(dirname(__file__), "../models/target", name)
    model.save(filePath)


def train_model(model: Sequential, dataset: Dataset, hyperpar: Dict):
    epochs = hyperpar["epochs"]
    learningRate = hyperpar["learningRate"]
    batchSize = hyperpar["batchSize"]

    optimizer = keras.optimizers.Adam(learning_rate=learningRate, name="Adam")
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.CategoricalAccuracy()]

    model.compile(optimizer, loss, metrics)
    dataset = dataset.batch(batchSize, drop_remainder=True)
    return model.fit(dataset, epochs=epochs)


def evaluate_model(model: Sequential, dataset: Dataset):
    batchSize = 100
    dataset = dataset.batch(batchSize, drop_remainder=True)
    return model.evaluate(dataset)
