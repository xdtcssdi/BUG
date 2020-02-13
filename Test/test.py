import numpy as np

from util import load_CIFAR10

if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = load_CIFAR10(r"/Users/oswin/Documents/BUG/datasets/cifar-10-python")
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)