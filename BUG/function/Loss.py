import numpy as np


class CrossEntry:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, Y_train, Y_hat):
        m = Y_train.shape[0]
        cost = - Y_train * np.log(np.clip(Y_hat, self.epsilon, 1.0-self.epsilon)) - \
               (1 - Y_train) * np.log(np.clip(1 - Y_hat, self.epsilon, 1.0-self.epsilon))
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
        cost = - targets * np.log(np.clip(outputs, self.epsilon, 1.0-self.epsilon)) - \
               (1 - targets) * np.log(np.clip(1 - outputs, self.epsilon, 1.0-self.epsilon))
        J = np.mean(np.sum(cost, axis=1, keepdims=True), axis=0)
        return np.squeeze(J)

    def backward(self, targets, outputs):
        return -targets / np.clip(outputs, self.epsilon, 1.0-self.epsilon) + \
               (1 - targets) / np.clip(1 - outputs, self.epsilon, 1.0-self.epsilon)

