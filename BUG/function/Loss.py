from BUG.load_package import p


class CrossEntry:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, Y_train, Y_hat):
        m = Y_train.shape[0]
        p.clip(Y_hat, self.epsilon, 1.0 - self.epsilon, out=Y_hat)
        cost = - Y_train * p.log(Y_hat) - (1 - Y_train) * p.log(1 - Y_hat)
        J = p.mean(cost, axis=0, keepdims=True)
        return p.squeeze(J)

    def backward(self, Y_train, Y_hat):
        p.clip(Y_hat, self.epsilon, 1.0 - self.epsilon, out=Y_hat)
        return -Y_train / Y_hat + (1 - Y_train) / (1 - Y_hat)


class SoftCategoricalCross_entropy:
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, targets, outputs):
        p.clip(outputs, self.epsilon, 1.0 - self.epsilon, out=outputs)

        if targets.ndim == 1 and outputs.ndim == 2:
            N, T = outputs.shape
            loss = -p.sum(p.log(outputs[p.arange(N), targets])) / N
            return loss
        N, T = targets.shape
        if outputs.ndim == 3:
            N, T, D = outputs.shape
            outputs = outputs.reshape(N * T, D)
        if targets.ndim == 3:
            N, T, D = targets.shape
            targets = targets.reshape(N * T, D)

        loss = -p.sum(p.log(outputs[p.arange(N * T), targets.reshape(N * T, )])) / N
        return loss

    def backward(self, targets, outputs):
        """
        :param targets:
        :param outputs: shape=(N,1)
        :return:
        """
        if outputs.shape[-1] != targets.shape[-1]:
            N, T = targets.shape
            targets = targets.reshape(N * T)
            dx_flat = outputs.copy()
            dx_flat[p.arange(N * T), targets] -= 1
            dx_flat /= N
            dx = dx_flat.reshape(N, T, -1)
            return dx
        dx_flat = outputs.copy()
        dx_flat[range(outputs.shape[0]), targets] -= 1
        return dx_flat
