from numpy.testing import assert_equal
import shadow_data as sd
import datasets as ds
import numpy as np
import pytest
import itertools
sd.set_seed(1234)
# Magic numbers (and arrays) come from seeded RNG, should hopefully be portable


class TestHillClimbing():
    def test__generate_labels(self):
        size = 10
        a = sd._generate_labels(4, size)
        b = np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3]).reshape((size, 1))
        assert_equal(a, b)

        size = 8
        a = sd._generate_labels(3, size)
        b = np.array([0, 0, 0, 1, 1, 1, 2, 2]).reshape((size, 1))
        assert_equal(a, b)

        size = 15
        a = sd._generate_labels(5, size)
        b = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3,
                     3, 4, 4, 4]).reshape((size, 1))
        assert_equal(a, b)

    def test__randomize_features(self):
        a = np.repeat(0, 10).reshape((1, 10))
        b = sd._randomize_features(a, 2, numFeatures=10)

        assert b[0, 1] == 1
        assert b[0, 7] == 1

        a = np.repeat(0, 100).reshape((1, 100))
        b = sd._randomize_features(a, 3, numFeatures=100)

        assert b[0, 0] == 1
        assert b[0, 11] == 1
        assert b[0, 74] == 1

    def test__get_random_record(self):
        a = sd._get_random_record(20)
        b = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
                     0, 1, 0, 1, 1, 1, 0]).reshape((1, 20))
        assert_equal(a, b)

    def test__randomize_features_batched(self):
        gen = np.random.default_rng(1234)
        gen2 = np.random.default_rng(1234)
        x = sd._get_random_record(600, gen)
        x2 = sd._get_random_record(600, gen)
        y = sd._get_random_records(600, 2, gen2)
        assert_equal(x[0], y[0])
        assert_equal(x2[0], y[1])


class TestNoisy():
    def test_generate_shadow_data_noisy_size(self):
        data = ds.load_dataset("kaggle")
        inputSize = data.cardinality().numpy()

        size = 100
        noisy = sd.generate_shadow_data_noisy(data, size, 0.1)
        assert noisy.cardinality().numpy() == size

        size = inputSize
        noisy = sd.generate_shadow_data_noisy(data, size, 0.1)
        assert noisy.cardinality().numpy() == size

        size = inputSize * 2
        noisy = sd.generate_shadow_data_noisy(data, size, 0.1)
        assert noisy.cardinality().numpy() == size

        size = inputSize * 2 + 3
        noisy = sd.generate_shadow_data_noisy(data, size, 0.1)
        assert noisy.cardinality().numpy() == size

    @pytest.mark.skip("Takes long.")
    def test_generate_shadow_data_noisy_content(self):
        # TODO: Since datasets are shuffled this test somewhat trivial
        data = ds.load_dataset("kaggle")
        inputSize = data.cardinality().numpy()
        compSize = 2
        data_subset = data.take(compSize)
        def ar(x): return x[0].numpy()

        size = 100
        noisy = sd.generate_shadow_data_noisy(data, size, 0.1).take(compSize)
        for a, b in itertools.zip_longest(data_subset, noisy):
            with pytest.raises(AssertionError):
                assert_equal(ar(a), ar(b))

        size = inputSize
        noisy = sd.generate_shadow_data_noisy(data, size, 0.1).take(compSize)
        for a, b in itertools.zip_longest(data_subset, noisy):
            with pytest.raises(AssertionError):
                assert_equal(ar(a), ar(b))

        size = inputSize * 2
        noisy = sd.generate_shadow_data_noisy(data, size, 0.1).take(compSize)
        for a, b in itertools.zip_longest(data_subset, noisy):
            with pytest.raises(AssertionError):
                assert_equal(ar(a), ar(b))

        size = inputSize * 2 + 3
        noisy = sd.generate_shadow_data_noisy(data, size, 0.1).take(compSize)
        for a, b in itertools.zip_longest(data_subset, noisy):
            with pytest.raises(AssertionError):
                assert_equal(ar(a), ar(b))
