from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

from os.path import dirname, join

from tensorflow import keras
from tensorflow.data import Dataset
from tensorflow.keras.activations import tanh
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPool2D, Softmax
from tensorflow.keras.models import Sequential

global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed


class Model:

    epochs = 1  # Overridden by subclass
    learningRate = 0.01
    learningRateDecay = int(1e-07)
    activation = tanh
    batchSize = 100
    optimizer = keras.optimizers.Adam(learning_rate=learningRate, name="Adam")
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [keras.metrics.CategoricalAccuracy()]

    def __init__(self, name: str) -> None:
        self.name = name
        self.filePath = join(dirname(__file__), "../models", name)
        print(self.model.summary())

    def train(self, x_train: Dataset, y_train: Dataset):
        self.model.compile(self.optimizer, self.loss, self.metrics)
        return self.model.fit(x_train, y_train, self.batchSize, self.epochs)

    def save(self):
        self.model.save(self.filePath)

    def load(self):
        self.model = keras.models.load_model(self.filePath)


class CifarModel(Model):
    """
    On CIFAR datasets, we train a standard convolutional neural
    network (CNN) with two convolution and max pooling layers
    plus a fully connected layer of size 128 and a Sof tMax layer.
    ‘We use Tanh as the activation function. We set the learning
    rate to 0.001, the learning rate decay to le — 07, and the
    maximum epochs of training to 100.
    """

    epochs = 100

    def __init__(self, name: str) -> None:
        self.model = Sequential()
        self.model.add(Input((32, 32, 3)))
        self.model.add(Conv2D(3, 2, activation=self.activation))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(3, 2, activation=self.activation))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dense(128, activation=self.activation))
        self.model.add(Softmax())
        super().__init__(name)


class KaggleModel(Model):
    """
    On the purchase dataset (see Section VI-A), we train a fully
    connected neural network with one hidden layer of size 128
    and a SoftMax layer. We use Tanh as the activation function.
    ‘We set the learning rate to 0.001, the learning rate decay to
    1e — 07, and the maximum epochs of training to 200.
    """
    epochs = 200

    def __init__(self, name: str) -> None:
        self.model = Sequential()
        self.model.add(Input((600, 1)))
        self.model.add(Dense(128, activation=self.activation))
        self.model.add(Softmax())
        super().__init__(name)
