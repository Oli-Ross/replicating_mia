"""
.. include:: ../docs/shadow_data.md
"""

# TODO: Everything (?) in here is Kaggle specific

from os import environ
from typing import Tuple
from numpy.typing import NDArray
from typing import Optional, Dict, List
import datasets as ds
import random
import numpy as np

# Tensorflow C++ backend logging verbosity
environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # NOQA

import tensorflow as tf
from tensorflow.keras.utils import to_categorical  # pyright: ignore
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


def split_shadow_data(config: Dict, shadowData: ds.Dataset) -> List[ds.Dataset]:
    print("Splitting shadow data into subsets.")
    numSubsets = config["shadowModels"]["number"]
    return ds.split_dataset(shadowData, numSubsets)


def load_shadow_data(config: Dict):
    dataName = get_shadow_data_name(config)
    return ds.load_shadow(dataName, verbose=config["verbose"])


def get_shadow_data_name(config: Dict):
    shadowConfig = config["shadowDataset"]
    method = shadowConfig["method"]
    targetDataName = config["targetDataset"]["name"]
    dataSize = shadowConfig["size"]
    hyperpars = shadowConfig[method]["hyperparameters"]
    if method == "noisy":
        dataName = f'{method}_fraction_{hyperpars["fraction"]}_size_{dataSize}_target_{targetDataName}'
    elif method == "hill_climbing":
        dataName = \
            f'{method}_' + \
            f'{targetDataName}_' + \
            f'kmax_{hyperpars["k_max"]}_' + \
            f'kmin_{hyperpars["k_min"]}_' + \
            f'confmin_{hyperpars["conf_min"]}_' + \
            f'rejmax_{hyperpars["rej_max"]}_' + \
            f'itermax_{hyperpars["iter_max"]}_' + \
            f'size_{dataSize}'
    elif method == "original":
        dataName = \
            f'{method}_' + \
            f'{targetDataName}_' + \
            f'original_data_' + \
            f'size_{dataSize}'
    elif method == "statistic":
        dataName = \
            f'{method}_' + \
            f'{targetDataName}_' + \
            f'statistic_' + \
            f'size_{dataSize}'
    else:
        raise ValueError(f"{method} is not a valid shadow data method.")
    return dataName


def get_shadow_data(config: Dict) -> ds.Dataset:
    verbose = config["verbose"]
    shadowConfig = config["shadowDataset"]
    method = shadowConfig["method"]
    dataSize = shadowConfig["size"]
    hyperpars = shadowConfig[method]["hyperparameters"]
    dataName = get_shadow_data_name(config)

    try:
        print("Loading shadow data from disk.")
        shadowData = load_shadow_data(config)
    except BaseException:
        print("Loading failed, generating shadow data.")

        targetDataset = get_target_model_rest_data(config)
        targetModel = tm.load_model(tm.get_model_name(config), verbose=config["verbose"])

        if method == "noisy":
            shadowData = generate_shadow_data_noisy(targetDataset, dataSize, **hyperpars)
        elif method == "hill_climbing":
            shadowData = hill_climbing(targetModel, dataSize, **hyperpars)
        elif method == "original":
            modelName = tm.get_model_name(config)
            restDataName = modelName + "_rest_data"
            shadowData = ds.load_target(restDataName).take(dataSize)
        elif method == "statistic":
            shadowData = generate_shadow_data_statistic(config)
        else:
            raise ValueError(f"{method} is not a valid shadow data method.")

        if verbose:
            print(f"Saving shadow data {dataName} to disk.")
        try:
            ds.save_shadow(shadowData, dataName)
        except BaseException:
            print(f"Failed to save shadow data {dataName} to disk.")
            ds.delete_shadow(dataName)
            raise

    return shadowData


def _make_data_record_noisy(features, label, fraction):
    # TODO: numFeatures is hardcoded
    numFeatures = 600
    k = int(numFeatures * fraction)
    return _randomize_features(features, k=k).reshape(numFeatures), label


def _make_dataset_noisy(original_data: Dataset, fraction: float) -> Dataset:
    """
    Returns new dataset, where each element has a fraction of its features
    flipped.
    """
    return original_data.map(
        lambda x, y:
            tf.numpy_function(func=_make_data_record_noisy, inp=(x, y, fraction), Tout=[tf.int64, tf.int64])
    )


def generate_shadow_data_noisy(original_data: Dataset, outputSize: int, fraction: float = 0.1) -> Dataset:
    """
    Generate synthetic data for the shadow models by using a noisy version of
    the original data.
    Returns only the noisy data, no the oririnal data.

    Arguments:
        fraction: percentage of labels that will be flipped per data record to
                  make it "noisy"
    """
    inputSize = original_data.cardinality().numpy()
    # Since outputSize % inputSize not always 0, we have to fill the gap with a subset
    # of the full input data. To avoid bias, shuffle the input data.
    noisySet = _make_dataset_noisy(ds.shuffle(original_data), fraction)

    if inputSize >= outputSize:
        return noisySet.take(outputSize)

    numNoisyVersions = int(np.floor(outputSize / inputSize))
    # How many records to add after collecting numNoisyVersions sets
    offset = outputSize % inputSize

    for _ in range(numNoisyVersions - 1):
        newNoisySet = _make_dataset_noisy(ds.shuffle(original_data), fraction)
        noisySet = noisySet.concatenate(newNoisySet)

    offsetSet = _make_dataset_noisy(
        ds.shuffle(original_data), fraction).take(offset)
    return noisySet.concatenate(offsetSet)

def _get_filter_fn(label: int):

    wantedLabel = np.int64(label)
    def _filter_fn(_, y): return tf.math.equal(wantedLabel, tf.math.argmax(y))
    return _filter_fn

def _compute_kaggle_marginals(config):
    dataName = config["targetDataset"]["name"]
    originalData = ds.load_dataset(dataName)
    numFeatures = iter(originalData).get_next()[0].numpy().shape[0]
    numLabels = iter(originalData).get_next()[1].numpy().shape[0]
    marginalProbabilities = np.zeros((100,600))
    # For each class, count binary feature values to get marginal
    for _class in range(numLabels):
        if config["verbose"]:
            print(f"Computing marginal probability for class {_class}/{numLabels}")
        filteredData = originalData.filter(_get_filter_fn(_class))
        initialCount = np.array([0]*numFeatures)
        countedFeatures = filteredData.reduce(initialCount, lambda oldCount, dataPoint: oldCount + dataPoint[0]).numpy()
        sampleSize = filteredData.cardinality()
        if sampleSize <= 0 and config["verbose"]:  # tf uses constants < 0 to indicate unknown cardinality
            sampleSize = len(list(filteredData.as_numpy_iterator()))
        marginalProbability = countedFeatures / sampleSize
        marginalProbabilities[_class] = marginalProbability
    return marginalProbabilities

def generate_shadow_data_statistic(config: Dict) -> Dataset:
    """
    Generate synthetic data for the shadow models by using the marginal
    distribution of features in the original dataset.
    """
    # TODO: Kaggle specific
    size = config["shadowDataset"]["size"]
    try:
        marginalProbabilities = ds.load_numpy_array("kaggle_marginals.npy")
    except:
        marginalProbabilities = _compute_kaggle_marginals(config)
        ds.save_numpy_array("kaggle_marginals.npy",marginalProbabilities)

    # Generate new records
    numClasses = marginalProbabilities.shape[0]
    numFeatures = marginalProbabilities.shape[1]

    features: NDArray = np.zeros((size,numFeatures)).astype(np.int32)
    labels: NDArray = np.zeros((size,numClasses)).astype(np.int32)

    recordsPerClass = int(size/numClasses)

    for _class in range(numClasses):
        if config["verbose"]:
            print(f"Generating records for class {_class}")
        index_start = _class * recordsPerClass
        index_end = (_class + 1) * recordsPerClass
        #  for index in range(index_start, index_end):
        #      labels[index] = to_categorical(_class, num_classes = numClasses)
        labels[index_start:index_end] = np.tile(to_categorical(_class, num_classes =
                                                               numClasses),recordsPerClass).reshape(recordsPerClass,numClasses)
        gen = np.random.default_rng(seed=global_seed)
        marginalProbability = marginalProbabilities[_class]

        for feature in range(numFeatures):
            probability = marginalProbability[feature]
            # sample one feature for all records in this class at once
            sampledFeature = gen.choice([0,1], p=[1-probability,probability],size = recordsPerClass)
            features[index_start:index_end,feature] = sampledFeature

    shadowData = Dataset.from_tensor_slices((features, labels))
    shadowData = shadowData.shuffle(size, seed=global_seed, reshuffle_each_iteration=False)
    return shadowData




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


def _randomize_features(data: NDArray, k: int,
                        numFeatures: int = 600) -> NDArray:

    featuresToFlip = random.sample(range(numFeatures), k)

    data = data.reshape((1, numFeatures))

    data[0][featuresToFlip] ^= 1

    return data


def _get_random_record(numFeatures: int,
                       randomGenerator=globalRandomGen) -> NDArray:

    x = randomGenerator.integers(0, high=1, endpoint=True, size=numFeatures)

    return x.reshape((1, numFeatures))


def _randomize_features_batched(
        data: NDArray, k: int, batchSize: int, numFeatures: int = 600) -> NDArray:

    outputdata = np.repeat(data.reshape((numFeatures, 1)), batchSize, axis=1).transpose()

    import numpy.testing as tt
    tt.assert_equal(outputdata[0], data.reshape(numFeatures))

    # Flip features of the first record
    featuresToFlip = random.sample(range(numFeatures), k)
    outputdata[0, featuresToFlip] ^= 1

    # Flip all further records based on the previous one
    for i in range(1,batchSize):
        featuresToFlip = random.sample(range(numFeatures), k)
        outputdata[i,:] = outputdata[i-1, :]
        outputdata[i, featuresToFlip] ^= 1

    return outputdata


def _rebatch(x, k, batchSize, targetModel) -> Tuple[NDArray, NDArray, int]:
    xs = _randomize_features_batched(x, k, batchSize)
    ys = targetModel.predict(xs, batch_size=batchSize, verbose=0)
    return xs, ys, 0


def _generate_synthetic_record_batched(label: int,
                                       targetModel: Sequential,
                                       k_max: int = 200,
                                       k_min: int = 5,
                                       conf_min: float = 0.05,
                                       rej_max: int = 20,
                                       iter_max: int = 200,
                                       batchSize: int = 1) -> Optional[NDArray]:
    """
    Synthesize a data record, using Algorithm 1 from Shokri et als
    paper "Membership Inference Attacks against Machine Learning Models".
    """
    assert label < 100 and label >= 0

    # Initalization
    batchIndex: int = 0
    numFeatures: int = 600
    kWasUpdated = False
    k = k_max
    y_c_star = 0
    j = 0
    x = _get_random_record(numFeatures)
    haveSampled = False

    if batchSize == 1:
        xs = x.reshape((1, 600))
        ys = targetModel.predict(xs, batch_size=batchSize, verbose=0)
    else:
        xs, ys, batchIndex = _rebatch(x, k, batchSize, targetModel)

    # Controls number of iterations
    for i in range(iter_max):

        x = xs[batchIndex]
        y = ys[batchIndex]
        y_c = y[label]
        predictedClass = np.argmax(y, axis=0)

        if y_c >= y_c_star:
            if y_c > conf_min and predictedClass == label:
                #  print(f"Now sampling! {batchIndex},{y_c},{y_c_star}")
                haveSampled = True
                if y_c > globalRandomGen.random():
                    return x.reshape((1, numFeatures))

            xs, ys, batchIndex = _rebatch(x, k, batchSize, targetModel)
            y_c_star = y_c
            j = 0
            continue
        else:
            j = j + 1
            if j > rej_max and (k != k_min) and haveSampled:
                k = int(max(k_min, np.ceil(k / 2)))
                j = 0
                kWasUpdated = True

        batchExhausted = (batchIndex == batchSize - 1)

        if batchExhausted or kWasUpdated:
            xs, ys, batchIndex = _rebatch(x, k, batchSize, targetModel)
            kWasUpdated = False
        else:
            batchIndex += 1

        #  if (i % 20) == 0:
        #      print(f"{i}/{iter_max}, y_c/y_c*: {y_c:.1%}/{y_c_star:.1%}, pred/class: {predictedClass}/{label}")

    return None


def _generate_synthetic_record(label: int,
                               targetModel: Sequential,
                               k_max: int = 200,
                               k_min: int = 5,
                               conf_min: float = 0.05,
                               rej_max: int = 20,
                               iter_max: int = 200,
                               batchSize: int = 1) -> Optional[NDArray]:
    """
    Synthesize a data record, using Algorithm 1 from Shokri et als
    paper "Membership Inference Attacks against Machine Learning Models".
    """
    assert label < 100 and label >= 0

    # Initalization
    numFeatures: int = 600
    k = k_max
    y_c_star = 0
    j = 0
    x = _get_random_record(numFeatures)

    # Controls number of iterations
    for i in range(iter_max):

        y = targetModel.predict(x, batch_size=1, verbose=0)
        y_c = y[0][label]
        predictedClass = np.argmax(y, axis=1)[0]

        if y_c >= y_c_star:
            if y_c > conf_min and predictedClass == label:
                #  print("Now sampling!")
                if y_c > globalRandomGen.random():
                    return x

            y_c_star = y_c
            j = 0
        else:
            j = j + 1
            if j > rej_max and (k != k_min):
                k = int(max(k_min, np.ceil(k / 2)))
                j = 0

        x = _randomize_features(x, k)  # pyright: ignore

        #  if (i % 20) == 0:
        #      print(
        #          f"{i}/{iter_max}, y_c/y_c*: {y_c:.1%}/{y_c_star:.1%}, pred/class: {predictedClass}/{label}")

    return None


def hill_climbing(targetModel: Sequential, numRecords: int,
                  **hyperpars) -> Dataset:
    """
    Generate synthetic data for the shadow models by querying the target model
    for randomly sampled records, in order to find those that are classified
    with high confidence.

    `numRecords`: size of generated dataset
    `hyperpars` has the following keys (taken from the paper:
    k_max,k_min,rej_max,conf_min,iter_max)
    """

    # Generate an array of labels, determining which class to synthesize for
    # TODO: initializing and then changing `features` array might not be most
    # efficient solution

    numClasses: int = 100
    labels: NDArray = _generate_labels(numClasses, numRecords)

    numFeatures: int = 600
    features: NDArray = np.zeros((numRecords, numFeatures))

    for index, label in enumerate(labels):
        label = int(label[0])
        new_record = _generate_synthetic_record(label, targetModel, **hyperpars)
        while new_record is None:
            new_record = _generate_synthetic_record(label, targetModel, **hyperpars)
        print(f"Generating synthetic records: {index}/{numRecords}, {index/numRecords*100:.2f}% done.")
        features[index] = new_record.reshape((1, numFeatures))

    features = features.reshape((numRecords, numFeatures))
    labels = labels.reshape((numRecords, 1))
    return Dataset.from_tensor_slices((features, labels))

def get_target_model_rest_data(config:Dict) -> Dataset:
    modelName = tm.get_model_name(config)
    restDataName = modelName + "_rest_data"
    return ds.load_target(restDataName)

if __name__ == "__main__":
    import argparse
    import configuration as con
    import datasets as ds
    import target_models as tm

    parser = argparse.ArgumentParser(description='Generate all the necessary shadow data and save it to disk.')
    parser.add_argument('--config', help='Relative path to config file.',)
    config = con.from_cli_options(vars(parser.parse_args()))
    set_seed(config["seed"])

    shadowData = get_shadow_data(config)
