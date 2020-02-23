import numpy as np


class CrossEntry:
    def forward(self, Y_train, Y_hat):
        m = Y_train.shape[0]
        cost = - Y_train * np.log(np.clip(Y_hat, 1e-8, 1.0)) - (1 - Y_train) * np.log(np.clip(1 - Y_hat, 1e-8, 1.0))
        J = np.mean(cost, axis=0, keepdims=True)
        # if regularization_mode == 'L1':
        #     J += lambd/m * np.linalg.norm()
        return np.squeeze(J)

    def backward(self, Y_train, Y_hat):
        return -Y_train / np.clip(Y_hat, 1e-8, 1.0) + (1 - Y_train) / np.clip(1 - Y_hat, 1e-8, 1.0)


class SoftCategoricalCross_entropy:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, targets, outputs):
        outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        J = np.mean(-np.sum(targets * np.log(outputs), axis=1, keepdims=True), axis=0)
        return np.squeeze(J)

    def backward(self, targets, outputs):
        outputs = np.clip(outputs, self.epsilon, 1 - self.epsilon)
        return outputs - targets
