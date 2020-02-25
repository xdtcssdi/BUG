import cupy as cp


class CrossEntry:
    def forward(self, Y_train, Y_hat):
        m = Y_train.shape[0]
        cost = - Y_train * cp.log(cp.clip(Y_hat, 1e-8, 1.0)) - (1 - Y_train) * cp.log(cp.clip(1 - Y_hat, 1e-8, 1.0))
        J = cp.mean(cost, axis=0, keepdims=True)
        # if regularization_mode == 'L1':
        #     J += lambd/m * np.linalg.norm()
        return cp.squeeze(J)

    def backward(self, Y_train, Y_hat):
        return -Y_train / cp.clip(Y_hat, 1e-8, 1.0) + (1 - Y_train) / cp.clip(1 - Y_hat, 1e-8, 1.0)


class SoftCategoricalCross_entropy:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, targets, outputs):
        outputs = cp.clip(outputs, self.epsilon, 1 - self.epsilon)
        cost = - targets * cp.log(cp.clip(outputs, self.epsilon, 1.0-self.epsilon)) - \
               (1 - targets) * cp.log(cp.clip(1 - outputs, self.epsilon, 1.0-self.epsilon))
        J = cp.mean(cp.sum(cost, axis=1, keepdims=True), axis=0)
        return cp.squeeze(J)

    def backward(self, targets, outputs):
        return -targets / cp.clip(outputs, self.epsilon, 1.0-self.epsilon) + (1 - targets) / cp.clip(1 - outputs, self.epsilon, 1.0-self.epsilon)
