from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPool2D, Softmax
from tensorflow.keras.models import Sequential


class Model:
    def __init__(self) -> None:
        print(self.model.summary())

    def train(self):
        pass


class CifarModel(Model):
    """
    On CIFAR datasets, we train a standard convolutional neural
    network (CNN) with two convolution and max pooling layers
    plus a fully connected layer of size 128 and a Sof tMax layer.
    ‘We use Tanh as the activation function. We set the learning
    rate to 0.001, the learning rate decay to le — 07, and the
    maximum epochs of training to 100.
    """

    def __init__(self) -> None:
        self.model = Sequential()
        self.model.add(Input((32, 32, 3)))
        self.model.add(Conv2D(3, 2))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Conv2D(3, 2))
        self.model.add(MaxPool2D(pool_size=(2, 2)))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Softmax())
        super().__init__()


class KaggleModel(Model):
    """
    On the purchase dataset (see Section VI-A), we train a fully
    connected neural network with one hidden layer of size 128
    and a SoftMax layer. We use Tanh as the activation function.
    ‘We set the learning rate to 0.001, the learning rate decay to
    1e — 07, and the maximum epochs of training to 200.
    """

    def __init__(self) -> None:
        self.model = Sequential()
        self.model.add(Input((600, 1)))
        self.model.add(Dense(128, activation='tanh'))
        self.model.add(Softmax())
        super().__init__()
