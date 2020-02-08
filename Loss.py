import numpy as np


class CrossEntry:
    def forward(self, Y_train, Y_hat):
        cost = - Y_train * np.log(Y_hat) - (1 - Y_train) * np.log(1 - Y_hat)
        J = np.mean(cost, axis=1, keepdims=True)
        return np.squeeze(J)

    def backward(self, Y_train, Y_hat):
        return -Y_train / Y_hat + (1 - Y_train) / (1 - Y_hat)


class SoftCategoricalCross_entropy:
    def forward(self, Y_train, Y_hat):
        return -np.mean(Y_train * np.log(1 - Y_hat), axis=1)

    def backward(self, Y_train, Y_hat):
        return Y_train - Y_hat