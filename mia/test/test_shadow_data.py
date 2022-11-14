from numpy.testing import assert_equal
import shadow_data as sd
import numpy as np
sd.set_seed(1234)


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
        print(b)

        assert b[0, 0] == 1
        assert b[0, 11] == 1
        assert b[0, 74] == 1
        # Magic numbers in the assert statements come from seeded RNG
