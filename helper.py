import numpy as np


def W_random_init(shape, C=1):
    return np.random.randn(shape[0], shape[1]) * np.sqrt(C/shape[1])

