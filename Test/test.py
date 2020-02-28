import os

import numpy as np

from BUG.Layers.Layer import RNN

def f(*aa):
    print(aa)
    aa[0][0] *= 100

if __name__ == '__main__':
    a = np.array([1])

    path = 'aaa'
