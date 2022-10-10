"""
.. include:: ../docs/shadow_data.md
"""

from os import environ
from typing import Tuple
from numpy.typing import NDArray
import numpy as np

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras import Sequential

global_seed: int = 1234


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed


def generate_shadow_data_sampling(original_data: Dataset) -> Dataset:
    """
    Generate synthetic data for the shadow models by randomly sampling data
    points from the original data set.
    """
    sample_dataset: Dataset = tf.data.Dataset.sample_from_datasets(
        [original_data], seed=global_seed, stop_on_empty_dataset=True)
    return sample_dataset


def _make_data_record_noisy(features, labels):
    print(features.shape)
    print(labels.shape)
    # Do something to the data record
    return features, labels


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


def generate_shadow_data_model(target_model: Sequential) -> Dataset:
    """
    Generate synthetic data for the shadow models by querying the target model
    for randomly sampled records, in order to find those that are classified
    with high confidence.
    """
    pass


if __name__ == "__main__":
    import datasets
    kaggle: Dataset = datasets.load_dataset("kaggle").take(10)
    kaggle_noisy: Dataset = generate_shadow_data_noisy(kaggle)
    print(tf.data.DatasetSpec.from_value(kaggle_noisy))
    print(tf.data.DatasetSpec.from_value(kaggle))
    for x, y in zip(kaggle.as_numpy_iterator(),
                    kaggle_noisy.as_numpy_iterator()):
        assert np.all(x[1] == y[1])
        assert np.all(x[0] == y[0])
