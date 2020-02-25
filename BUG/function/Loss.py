from BUG.load_package import p

class CrossEntry:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, Y_train, Y_hat):
        m = Y_train.shape[0]
        cost = - Y_train * p.log(p.clip(Y_hat, self.epsilon, 1.0-self.epsilon)) - \
               (1 - Y_train) * p.log(p.clip(1 - Y_hat, self.epsilon, 1.0-self.epsilon))
        J = p.mean(cost, axis=0, keepdims=True)
        # if regularization_mode == 'L1':
        #     J += lambd/m * p.linalg.norm()
        return p.squeeze(J)

    def backward(self, Y_train, Y_hat):
        return -Y_train / p.clip(Y_hat, 1e-8, 1.0) + (1 - Y_train) / p.clip(1 - Y_hat, 1e-8, 1.0)


class SoftCategoricalCross_entropy:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, targets, outputs):
        cost = - targets * p.log(p.clip(outputs, self.epsilon, 1.0-self.epsilon)) - \
               (1 - targets) * p.log(p.clip(1 - outputs, self.epsilon, 1.0-self.epsilon))
        J = p.mean(p.sum(cost, axis=1, keepdims=True), axis=0)
        return p.squeeze(J)

    def backward(self, targets, outputs):
        return -targets / p.clip(outputs, self.epsilon, 1.0-self.epsilon) + \
               (1 - targets) / p.clip(1 - outputs, self.epsilon, 1.0-self.epsilon)

