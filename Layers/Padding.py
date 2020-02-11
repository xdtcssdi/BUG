import numpy as np


def ZeroPad(z, pad=0):
    return np.pad(z, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
