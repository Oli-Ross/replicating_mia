"""
.. include:: ../docs/attack_data.md
"""

from os import environ
from math import floor
from typing import List, Dict, Tuple

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import target_models as tm
import datasets as ds

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset  # pyright: ignore
from tensorflow.python.framework import random_seed
from tensorflow.keras import Sequential  # pyright: ignore
from tensorflow.keras.utils import to_categorical  # pyright: ignore

global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed
    np.random.seed(global_seed)
    random_seed.set_seed(global_seed)


def _prepare_subset(superset: Dataset, size: int,
                    model: Sequential, inTraining: bool):
    # TODO: hardcoded batch size
    batchSize: int = 1
    subset = superset.take(size).batch(batchSize)
    predictions = model.predict(subset, batch_size=batchSize)
    if inTraining:
        labels = np.ones(size)
    else:
        labels = np.zeros(size)
    return predictions, labels


def _get_filter_fn(label: int):

    wantedLabel = np.int64(label)
    def _filter_fn(_, y): return tf.math.equal(wantedLabel, tf.math.argmax(y))
    return _filter_fn


def from_target_data(targetTrainData: Dataset, targetTestData: Dataset,
                     targetModel: Sequential, label: int) -> Dataset:
    # TODO: don't hardcode dataset size
    # TODO assertions about disjoint sets, and equal set sizes
    # TODO: Make this understandable without my piece of paper
    targetTrainData = targetTrainData.filter(_get_filter_fn(label))
    targetTestData = targetTestData.filter(_get_filter_fn(label))

    # There are only limited data points per class, thus we use as many as we
    # can get and split them 80/20 for training
    dataSizePerSet: int = min(len(list(targetTrainData.as_numpy_iterator())), len(
        list(targetTestData.as_numpy_iterator())))

    splitFactor = 0.8
    attackTestSize = int((1 - splitFactor) * dataSizePerSet)
    attackTrainSize = int(splitFactor * dataSizePerSet)

    print(f"Train dataset size (train) for label {label}: {attackTrainSize}")

    halfAttackTrainSize: int = int(attackTrainSize / 2)
    halfAttackTestSize: int = int(attackTestSize / 2)

    APredictions, ALabels = _prepare_subset(targetTestData, halfAttackTrainSize, targetModel, False)
    BPredictions, BLabels = _prepare_subset(targetTrainData, halfAttackTrainSize, targetModel, True)
    CPredictions, CLabels = _prepare_subset(
        targetTrainData.skip(halfAttackTrainSize), halfAttackTestSize, targetModel, True)
    DPredictions, DLabels = _prepare_subset(
        targetTestData.skip(halfAttackTrainSize), halfAttackTestSize, targetModel, False)

    featuresTrain = np.append(APredictions, BPredictions, axis=0)
    featuresTest = np.append(CPredictions, DPredictions, axis=0)
    labelsTrain = to_categorical(np.append(ALabels, BLabels, axis=0))
    labelsTest = to_categorical(np.append(CLabels, DLabels, axis=0))

    attackTrainData = Dataset.from_tensor_slices((featuresTrain, labelsTrain))
    attackTestData = Dataset.from_tensor_slices((featuresTest, labelsTest))

    return attackTrainData, attackTestData


def load(config: Dict) -> List[Tuple[ds.Dataset, ds.Dataset]]:
    verbose = config["verbose"]
    numClasses = config["targetModel"]["classes"]
    numDatasets = numClasses
    attackDatasets = []
    for i in range(numDatasets):
        testData = ds.load_attack(_get_attack_data_name(config, i, test=True), verbose=verbose)
        trainData = ds.load_attack(_get_attack_data_name(config, i, test=False), verbose=verbose)
        attackDatasets.append((testData, trainData))
    return attackDatasets


def _get_attack_data_name(config: Dict, i, test=False):
    numModels: int = config["shadowModels"]["number"]
    numClasses = config["targetModel"]["classes"]
    split: float = config["shadowModels"]["split"]
    if config["attackDataset"]["balance"]:
        balanced = "balanced"
    else:
        balanced = "unbalanced"
    name = tm.get_model_name(config) + f"_split_{split}_with_{numModels}_models_{i}_of_{numClasses}_{balanced}"
    if test:
        return name + "_test"
    else:
        return name + "_train"


def save(config: Dict, datasets: List[ds.Dataset]):
    numClasses = config["targetModel"]["classes"]
    assert numClasses == len(
        datasets), "List should contain 1 dataset per class"
    for index, (trainData, testData) in enumerate(datasets):
        if index % 10 == 0:
            print(f"Saving attack dataset #{index}/{numClasses}")
        ds.save_attack(trainData, _get_attack_data_name(config, index, test=False))
        ds.save_attack(testData, _get_attack_data_name(config, index, test=True))


def shuffle(dataset: Dataset, bufferSize=10000) -> Dataset:
    # TODO: hard coded buffer size (needs to be greater than dataset size for
    # real random sampling.)
    return dataset.shuffle(bufferSize, seed=global_seed, reshuffle_each_iteration=False)


def _balance_attack_data(dataset: ds.Dataset) -> ds.Dataset:
    in_data = dataset.filter(lambda _, y: tf.math.equal(tf.argmax(y), 1))
    out_data = dataset.filter(lambda _, y: tf.math.equal(tf.argmax(y), 0))

    in_points = len(list(in_data))
    out_points = len(list(out_data))

    if in_points > out_points:
        in_data = shuffle(in_data).skip(in_points - out_points)
    elif in_points < out_points:
        out_data = shuffle(out_data).skip(out_points - in_points)
    else:
        return dataset

    return in_data.concatenate(out_data)


def balance_attack_data(datasets: List[ds.Dataset]) -> List[ds.Dataset]:
    """
    Make sure that input datasets have equal number of in/out datapoints.
    """
    size = len(datasets)
    for index, dataset in enumerate(datasets):
        print(f"Balancing dataset {index} of {size}.")
        datasets[index] = _balance_attack_data(dataset)
    return datasets


def split_dataset(dataset: ds.Dataset, split: float):
    datasetSize = len(list(dataset))
    trainSize = floor(split * datasetSize)
    testSize = floor((1 - split) * datasetSize)
    assert trainSize + testSize <= datasetSize

    trainData = dataset.take(trainSize)
    testData = dataset.skip(trainSize).take(testSize)
    return trainData, testData


def split_attack_data_for_training(datasets: List[ds.Dataset], config: Dict):
    split = config["attackDataset"]["split"]
    splitDatasets = []
    for index, dataset in enumerate(datasets):
        if config["verbose"]:
            print(f"Splitting dataset {index}")
        splitDatasets.append(split_dataset(dataset, split))
    return splitDatasets


def get_attack_data(config: Dict,
                    shadowModels: List[tm.Sequential],
                    shadowDatasets: List[Tuple[ds.Dataset, ds.Dataset]]) -> List[Tuple[ds.Dataset, ds.Dataset]]:
    """
    This function predicts and then labels the provided datasets on their
    respective shadow model, thus creating the labeled data needed for the
    attack model.
    """
    try:
        print("Loading attack data.")
        return load(config)
    except BaseException:
        print("Didn't work, reconstructing it.")
        attackDatasets = from_shadow_models(config, shadowModels, shadowDatasets)
        balanceAttackData = config["attackDataset"]["balance"]
        if balanceAttackData:
            print("Balancing attack data to contain equal amounts in/out records.")
            attackDatasets = balance_attack_data(attackDatasets)
        print("Splitting attack data for training.")
        attackDatasets = split_attack_data_for_training(attackDatasets, config)
        print("Saving attack data to disk.")
        save(config, attackDatasets)
        return attackDatasets


def from_shadow_models(config: Dict, shadowModels:
                       List[tm.Sequential], shadowDatasets:
                       List[Tuple[ds.Dataset, ds.Dataset]]) -> List[ds.Dataset]:
    """
    Predicts the shadow data on the shadow models themselves and labels it with
    "in" and "out", for the attack model to train on.
    """
    numModels: int = config["shadowModels"]["number"]
    numClasses = config["targetModel"]["classes"]
    attackDatasets = []

    for i in range(numModels):

        model = shadowModels[i]
        trainData, testData = shadowDatasets[i]
        trainDataSize = trainData.cardinality().numpy()
        testDataSize = testData.cardinality().numpy()

        # Only relevant if split > 0.5
        assert trainDataSize >= testDataSize
        trainData = trainData.take(testDataSize)
        trainDataSize = testDataSize

        # Get predictions
        trainPreds = model.predict(trainData.batch(100, drop_remainder=False))
        testPreds = model.predict(testData.batch(100, drop_remainder=False))

        # Construct "in"/"out" labels
        trainLabels = np.tile(np.array([[1, 0]]), (trainDataSize, 1))
        testLabels = np.tile(np.array([[0, 1]]), (testDataSize, 1))

        # Combine them into 1 dataset
        trainPredsLabels = tf.data.Dataset.from_tensor_slices((trainPreds, trainLabels))
        testPredsLabels = tf.data.Dataset.from_tensor_slices((testPreds, testLabels))

        # Add data records and ground truth class to the dataset
        trainDataPredsLabels = tf.data.Dataset.zip((trainData, trainPredsLabels))
        testDataPredsLabels = tf.data.Dataset.zip((testData, testPredsLabels))

        # Combine train and test data
        attackData = trainDataPredsLabels.concatenate(testDataPredsLabels)

        for currentClass in range(numClasses):

            def is_current_class(dataAndClass, predAndLabel):
                (_, classLabel) = dataAndClass
                return tf.math.equal(np.int64(currentClass), tf.math.argmax(classLabel))

            classAttackData = attackData.filter(is_current_class)

            def restructure_data(dataAndClass, predAndLabel):
                return predAndLabel

            # Drop unused data record and class ground truth
            classAttackDataFinal = classAttackData.map(restructure_data)

            if i == 0:
                # First shadow model -> Each class seen the first time
                attackDatasets.append(classAttackDataFinal)
            else:
                # Not first shadow model. Concatenate with appropriate dataset
                attackDatasets[currentClass] = attackDatasets[currentClass].concatenate(classAttackDataFinal)

    return attackDatasets
