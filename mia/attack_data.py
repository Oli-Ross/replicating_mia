"""
.. include:: ../docs/attack_data.md
"""

from os import environ

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical


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
    def _filter_fn(x, y): return tf.math.equal(wantedLabel, tf.math.argmax(y))
    return _filter_fn


def from_target_data(targetTrainData: Dataset,
                     targetTestData: Dataset, targetModel: Sequential) -> Dataset:
    # TODO: don't hardcode dataset size
    # TODO assertions about disjoint sets, and equal set sizes
    # TODO: Make this understandable without my piece of paper
    label = 2
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

    APredictions, ALabels = _prepare_subset(
        targetTestData, halfAttackTrainSize, targetModel, False)
    BPredictions, BLabels = _prepare_subset(
        targetTrainData, halfAttackTrainSize, targetModel, True)
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

    # TODO: sort the data by its ground truth and only return partitions

    return attackTrainData, attackTestData
