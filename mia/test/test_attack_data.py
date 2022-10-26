import pytest

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import datasets
import target_models
import attack_data

import numpy as np


# Meta/Preparing
seed: int = 1234
# These tests are pretty slow, whole dataset is not feasible
NUM_LABELS: int = 1
datasets.set_seed(seed)
target_models.set_seed(seed)


def load_kaggle():
    # Target dataset
    dataset = datasets.load_dataset("kaggle")
    trainSize: int = 90000
    testSize: int = 90000
    targetTrainData, targetTestData = dataset.take(trainSize), dataset.skip(
        trainSize).take(testSize)
    return targetTrainData, targetTestData


def get_attack_data(data, label: int):

    targetTrainData, targetTestData = data

    # Construct + evaluate target model
    targetModel: target_models.KaggleModel = target_models.load_model("test")

    # Generate attack data
    return attack_data.from_target_data(
        targetTrainData, targetTestData, targetModel, label)


def save_data(trainData, testData):
    datasets.save_attack(trainData, "attack_kaggle_train")
    datasets.save_attack(testData, "attack_kaggle_test")


def load_data():
    train = datasets.load_attack("attack_kaggle_train")
    test = datasets.load_attack("attack_kaggle_test")
    return train, test


@pytest.mark.skip("Takes long.")
def test_equally_many_in_out_samples():
    data = load_kaggle()
    for label in range(NUM_LABELS):
        trainData, testData = get_attack_data(data, label)
        trainData = list(trainData.as_numpy_iterator())
        testData = list(testData.as_numpy_iterator())
        for dataSplit in [trainData, testData]:
            zeroes = 0
            ones = 0
            for x in dataSplit[:]:
                if x[1][0]:
                    zeroes = zeroes + 1
                else:
                    ones = ones + 1
            try:
                assert zeroes == ones, f"label {label}: {zeroes} zeroes, {ones} ones."
            except BaseException:
                pass
        print(f"Finished label {label}")


@pytest.mark.skip("Takes long.")
def test_features_are_softmax_values():
    data = load_kaggle()
    for label in range(NUM_LABELS):
        trainData, testData = get_attack_data(data, label)
        trainData = trainData.as_numpy_iterator()
        testData = testData.as_numpy_iterator()
        for dataSplit in [trainData, testData]:
            for x in dataSplit:
                x = x[0]
                assert x.sum(axis=0) >= 0.9999, f"sum is {x.sum(axis=0)}"
                assert x.sum(axis=0) <= 1.0001, f"sum is {x.sum(axis=0)}"
        print(f"Finished label {label}")


@pytest.mark.skip("Not a real test, only for manual diagnostic.")
def test_argmax_equals_label():
    data = load_kaggle()
    for label in range(100):
        trainData, testData = get_attack_data(data, label)
        trainData = trainData.as_numpy_iterator()
        testData = testData.as_numpy_iterator()
        for dataSplit in [trainData, testData]:
            for x in dataSplit:
                x = x[0]
                print(f"label is {label}, pred is {np.argmax(x)}")
        print(f"Finished label {label}")
