import numpy as np
import pytest
from scripts.data.utils import modcrop


@pytest.mark.parametrize(('array', 'scale', 'ground_truth'),
                         [(np.array(list(range(25))).reshape(5, 5),
                           2,
                           np.array([[0, 1, 2, 3],
                                     [5, 6, 7, 8],
                                     [10, 11, 12, 13],
                                     [15, 16, 17, 18]])),
                          (np.array(list(range(25))).reshape(5, 5),
                           3,
                           np.array([[0, 1, 2],
                                     [5, 6, 7],
                                     [10, 11, 12]])),
                          (np.array(list(range(25))).reshape(5, 5),
                           4,
                           np.array([[0, 1, 2, 3],
                                     [5, 6, 7, 8],
                                     [10, 11, 12, 13],
                                     [15, 16, 17, 18]]))])
def test_modcrop(array, scale, ground_truth):
    assert np.all(modcrop(array, scale) == ground_truth)
