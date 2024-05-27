import pytest
from util.points import *


def test_ensure_shape() :
    test_data = [(np.random.rand(3), 3, (3,1)),
                 (np.random.rand(3), 1, (1,3)),
                 (np.random.rand(1,3), 3, (3,1)),
                 (np.random.rand(1,3), 1, (1,3)),
                 (np.random.rand(7,6), 6, (6,7)),
                 (np.random.rand(7,6), 7, (7,6))]

    for x, d, shape in test_data :
        assert ensure_shape(x, d).shape == shape
