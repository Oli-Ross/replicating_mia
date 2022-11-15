"""
.. include:: ../docs/shadow_data.md
"""

# TODO: Everything (?) in here is Kaggle specific

from os import environ
from typing import Tuple
from numpy.typing import NDArray
from typing import Dict, Union
import random
import numpy as np

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import tensorflow as tf
from tensorflow.python.framework import random_seed
from tensorflow.data import Dataset  # pyright: ignore
from tensorflow.keras import Sequential  # pyright: ignore

global_seed: int = 1234
globalRandomGen = np.random.default_rng(global_seed)


def set_seed(new_seed: int):
    """
    Set the global seed that will be used for all functions that include
    randomness.
    """
    global global_seed
    global_seed = new_seed
    np.random.seed(global_seed)
    random.seed(global_seed)
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


def _randomize_features(data: NDArray, k: int, numFeatures: int = 600):

    featuresToFlip = random.sample(range(numFeatures), k)

    for index in featuresToFlip:
        data[0, index] = (data[0, index] + 1) % 2

    return data


def _get_random_record(numFeatures: int):
    x = np.repeat(1, numFeatures)
    for i in range(numFeatures):
        x[i] = globalRandomGen.integers(0, 1, endpoint=True)
    x = x.reshape((1, numFeatures))
    return x


def _generate_synthetic_record(
        label: int, targetModel: Sequential, hyperpars: Dict) -> Union[NDArray, None]:
    """
    Generate a synthesize data record, using Algorithm 1 from Shokri et als
    paper "Membership Inference Attacks against Machine Learning Models".
    """
    assert label < 100 and label >= 0

    # Initalization
    numFeatures: int = 600
    k = hyperpars["k_max"]
    k_min = hyperpars["k_min"]
    conf_min = hyperpars["conf_min"]
    rej_max = hyperpars["rej_max"]
    iter_max = hyperpars["iter_max"]
    y_c_star = 0
    j = 0
    x = _get_random_record(numFeatures)

    # Controls number of iterations
    for i in range(iter_max):

        y = targetModel.predict(x, batch_size=1)
        y_c = np.max(y, axis=1)[0]
        predictedClass = np.argmax(y, axis=1)[0]

        if y_c >= y_c_star:
            if y_c > conf_min and predictedClass == label:
                if y_c > globalRandomGen.random():
                    return x

            x_star = x
            y_c_star = y_c
            j = 0
        else:
            j = j + 1
            if j > rej_max:
                k = max(k_min, np.ceil(k / 2))
                j = 0
        x = _randomize_features(x_star, k)  # pyright: ignore
        print(f"Label {label}, got {predictedClass},y_c = {y_c}")

    return None


def hill_climbing(targetModel: Sequential, numRecords: int,
                  hyperpars: Union[Dict, None] = None) -> Dataset:
    """
    Generate synthetic data for the shadow models by querying the target model
    for randomly sampled records, in order to find those that are classified
    with high confidence.

    `numRecords`: size of generated dataset
    `hyperpars` has the following keys (taken from the paper:
    k_max,k_min,y_c_star,rej_max,conf_min)
    """
    if hyperpars is None:
        hyperpars = {"k_max": 200,
                     "k_min": 5,
                     "rej_max": 20,
                     "conf_min": 0.05,
                     "iter_max": 200}

    # Generate an array of labels, determining which class to synthesize for
    # TODO: initializing and then changing `features` array might not be most
    # efficient solution

    numClasses: int = 100
    labels: NDArray = _generate_labels(numClasses, numRecords)

    numFeatures: int = 600
    features: NDArray = np.zeros((numRecords, numFeatures))

    for index, label in enumerate(labels):
        label = int(label[0])
        new_record = _generate_synthetic_record(label, targetModel, hyperpars)
        while new_record is None:
            new_record = _generate_synthetic_record(
                label, targetModel, hyperpars)
        features[index] = new_record.reshape((1, numFeatures))

    features = features.reshape((numRecords, numFeatures))
    labels = labels.reshape((numRecords, 1))
    return Dataset.from_tensor_slices((features, labels))


if __name__ == "__main__":
    set_seed(1234)
    import target_models as tm
    model: tm.KaggleModel = tm.load_model(
        "lr_1e-3_bs_100_epochs_200_trainsize_10000")

    size = 2
    shadow_data = hill_climbing(model, size)
    for elem in shadow_data.as_numpy_iterator():
        print(elem[0].shape)
        print(elem[1].shape)
    model.predict(shadow_data, batch_size=size)
