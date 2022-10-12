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


def from_target_data(targetTrainData: Dataset,
                     targetTestData: Dataset, targetModel: Sequential) -> Dataset:
    # TODO: don't hardcode dataset size
    # TODO assertions about disjoint sets, and equal set sizes
    # TODO: Make this understandable without my piece of paper

    attackTrainSize: int = 10000
    attackTestSize: int = 2000
    halfAttackTrainSize: int = int(attackTrainSize / 2)
    halfAttackTestSize: int = int(attackTestSize / 2)

    A = targetTestData.take(halfAttackTrainSize).batch(100)
    B = targetTrainData.take(halfAttackTrainSize).batch(100)
    C = targetTrainData.skip(halfAttackTrainSize).take(
        halfAttackTestSize).batch(100)
    D = targetTestData.skip(halfAttackTrainSize).take(
        halfAttackTestSize).batch(100)

    APredictions = targetModel.predict(A, batch_size=100)
    BPredictions = targetModel.predict(B, batch_size=100)
    CPredictions = targetModel.predict(C, batch_size=100)
    DPredictions = targetModel.predict(D, batch_size=100)

    ALabels = np.zeros(halfAttackTrainSize)
    BLabels = np.ones(halfAttackTrainSize)
    CLabels = np.ones(halfAttackTestSize)
    DLabels = np.zeros(halfAttackTestSize)

    featuresTrain = np.append(
        APredictions,
        BPredictions,
        axis=0).astype(
        np.float64)
    featuresTest = np.append(
        CPredictions,
        DPredictions,
        axis=0).astype(
        np.float64)

    labelsTrain = np.append(ALabels, BLabels, axis=0).astype(np.int32)
    labelsTest = np.append(CLabels, DLabels, axis=0).astype(np.int32)
    labelsTrain = to_categorical(labelsTrain)
    labelsTest = to_categorical(labelsTest)
    attackTrainData = Dataset.from_tensor_slices(
        (featuresTrain, labelsTrain))
    attackTestData = Dataset.from_tensor_slices(
        (featuresTest, labelsTest))

    # TODO: sort the data by its ground truth and only return partitions

    return attackTrainData, attackTestData
