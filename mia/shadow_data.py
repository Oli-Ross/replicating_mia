"""
.. include:: ../docs/shadow_data.md
"""

# TODO: Everything (?) in here is Kaggle specific

from os import environ
from typing import Tuple
from numpy.typing import NDArray
from typing import Dict
import numpy as np

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import tensorflow as tf
from tensorflow.python.framework import random_seed
from tensorflow.data import Dataset  # pyright: ignore
from tensorflow.keras import Sequential  # pyright: ignore

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


def generate_shadow_data_sampling(original_data: Dataset) -> Dataset:
    """
    Generate synthetic data for the shadow models by randomly sampling data
    points from the original data set.
    """
    sample_dataset: Dataset = tf.data.Dataset.sample_from_datasets(
        [original_data], seed=global_seed, stop_on_empty_dataset=True)
    return sample_dataset


def _make_data_record_noisy(features, label):
    print(features.shape)
    print(label.shape)
    # Do something to the features
    return features, label


def generate_shadow_data_noisy(original_data: Dataset) -> Dataset:
    """
    Generate synthetic data for the shadow models by using a noisy version of
    the original data.
    """
    return original_data.map(
        lambda x, y: tf.numpy_function(
            func=_make_data_record_noisy,
            inp=(x, y),
            Tout=[tf.int64, tf.int64]))


def generate_shadow_data_statistic(original_data: Dataset) -> Dataset:
    """
    Generate synthetic data for the shadow models by using the marginal
    distribution of features in the original dataset.
    """
    pass


def _generate_labels(classes: int, size: int) -> NDArray:
    """
    Generate a numpy array of size `size`, where the values are integers between
    0 and `classes` - 1, distributed as evenly as possible.

    This array will be used to generate a synthetic array of features for each
    array element.
    """

    records_per_class: int = int(size / classes)
    extra_records: int = size % classes

    labels: NDArray = np.zeros((size, 1))
    index: int = 0

    for x in range(classes):
        if x < extra_records:
            records_for_this_class = records_per_class + 1
        else:
            records_for_this_class = records_per_class
        for y in range(records_for_this_class):
            labels[index + y, 0] = x
        index = index + records_for_this_class

    return labels


def _generate_synthetic_record(
        label: int, numFeatures: int, hyperpars: Dict) -> NDArray:
    """
    Generate a synthesize data record, using Algorithm 1 from Shokri et als
    paper "Membership Inference Attacks against Machine Learning Models".
    """

    assert label < 100 and label >= 0
    # TODO: Set desired class
    # TODO: Initialize record randomly
    # TODO: Query target model
    # TODO: Randomize some features

    # Placeholder
    features = np.repeat(label, numFeatures)
    features = features.reshape((1, numFeatures))
    return features


def hill_climbing(target_model: Sequential, numRecords: int,
                  hyperpars: Dict) -> Dataset:
    """
    Generate synthetic data for the shadow models by querying the target model
    for randomly sampled records, in order to find those that are classified
    with high confidence.

    `numRecords`: size of generated dataset
    """
    numClasses: int = hyperpars["numClasses"]
    numFeatures: int = hyperpars["numClasses"]

    # Generate an array of labels, determining which class to synthesize for
    labels: NDArray = _generate_labels(numClasses, numRecords)
    features: NDArray = np.zeros((numRecords, numFeatures))

    for index, label in enumerate(labels):
        features[index] = _generate_synthetic_record(label,
                                                     numFeatures, hyperpars)

    print(features)

    return Dataset.from_tensor_slices((features, labels))


if __name__ == "__main__":
    size = 10
    import target_models
    model: target_models.KaggleModel = target_models.load_model("test")
    my_shadow_data = hill_climbing(model, size=size)
    print(tf.data.DatasetSpec.from_value(my_shadow_data))
