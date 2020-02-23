import cupy as cp


def ZeroPad(z, pad=0):
    return cp.pad(z, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
