import numpy as np


def ZeroPad(Z, pad = 0):
    Z_pad = np.pad(Z, ((0, 0), (pad, pad), (pad, pad), (0, 0), ), 'constant')
    return Z_pad

