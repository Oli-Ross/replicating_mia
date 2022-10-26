import itertools

import datasets
import numpy as np
import pytest

COMPARE_SIZE = 2


def test_seed():
    datasets.set_seed(2222)

    assert datasets.global_seed == 2222


@pytest.mark.skip("Takes long.")
def test_shuffle():
    kaggle = datasets.load_dataset("kaggle")
    shuffled = datasets.shuffle_dataset(kaggle, 197324)
    kaggle = kaggle.take(COMPARE_SIZE).as_numpy_iterator()
    shuffled = shuffled.take(COMPARE_SIZE).as_numpy_iterator()

    for a, b in itertools.zip_longest(kaggle, shuffled):
        with pytest.raises(AssertionError):
            np.testing.assert_equal(a, b)


@pytest.mark.skip("Takes long.")
def test_deterministic_shuffling():
    kaggle = datasets.shuffle_dataset(datasets.load_dataset("kaggle"), 197324)
    kaggle_2 = datasets.shuffle_dataset(
        datasets.load_dataset("kaggle"), 197324)
    kaggle = kaggle.take(COMPARE_SIZE).as_numpy_iterator()
    kaggle_2 = kaggle_2.take(COMPARE_SIZE).as_numpy_iterator()

    for a, b in itertools.zip_longest(kaggle, kaggle_2):
        np.testing.assert_equal(a, b)


def test_load_dataset():
    cifar = datasets.load_dataset("cifar10")
    assert cifar.element_spec[0].shape == (32, 32, 3)
    assert cifar.element_spec[1].shape == ([10])

    cifar = datasets.load_dataset("kaggle")
    assert cifar.element_spec[0].shape == (600,)
    assert cifar.element_spec[1].shape == ([100])
