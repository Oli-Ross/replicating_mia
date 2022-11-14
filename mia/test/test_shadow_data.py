import shadow_data as sd
import numpy as np
from numpy.testing import assert_equal


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
