from BUG.load_package import p


class CrossEntry:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, Y_train, Y_hat):
        m = Y_train.shape[0]
        p.clip(Y_hat, self.epsilon, 1.0-self.epsilon, out=Y_hat)
        cost = - Y_train * p.log(Y_hat) - (1 - Y_train) * p.log(1 - Y_hat)
        J = p.mean(cost, axis=0, keepdims=True)
        # if regularization_mode == 'L1':
        #     J += lambd/m * p.linalg.norm()
        return p.squeeze(J)

    def backward(self, Y_train, Y_hat):
        p.clip(Y_hat, self.epsilon, 1.0-self.epsilon, out=Y_hat)
        return -Y_train / Y_hat + (1 - Y_train) /(1 - Y_hat )


class SoftCategoricalCross_entropy:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, targets, outputs):
        p.clip(outputs, self.epsilon, 1.0-self.epsilon, out=outputs)
        cost = - targets * p.log(outputs)
        J = p.mean(p.sum(cost, axis=1, keepdims=True), axis=0)
        return p.squeeze(J)

    def backward(self, targets, outputs):
        return p.clip(outputs - targets, self.epsilon, 1.0-self.epsilon)

