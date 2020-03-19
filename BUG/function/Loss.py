from BUG.load_package import p


class Loss:

    def forward(self, Y_train, Y_hat):
        raise NotImplementedError

    def backward(self, Y_train, Y_hat):
        raise NotImplementedError


class CrossEntry(Loss):
    def __init__(self, epsilon=1e-11):
        self.epsilon = epsilon

    def forward(self, Y_train, Y_hat):
        target = Y_train.reshape(Y_hat.shape)
        m = target.shape[0]
        p.clip(Y_hat, self.epsilon, 1.0 - self.epsilon, out=Y_hat)
        cost = - target * p.log(Y_hat) - (1 - target) * p.log(1 - Y_hat)
        J = p.sum(cost, axis=0, keepdims=True) / m
        return p.squeeze(J)

    def backward(self, Y_train, Y_hat):
        target = Y_train.reshape(Y_hat.shape)
        p.clip(Y_hat, self.epsilon, 1.0 - self.epsilon, out=Y_hat)
        return Y_hat - target


class SoftCategoricalCross_entropy(Loss):
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
        if outputs.shape[-1] != targets.shape[-1] and targets.ndim == 2:
            N, T = targets.shape
            targets = targets.reshape(N * T, )
            if outputs.ndim == 3:
                outputs = outputs.reshape(N*T, -1)
            dx_flat = outputs.copy()
            dx_flat[p.arange(N * T), targets] -= 1
            dx = dx_flat.reshape(N, T, -1)
            return dx
        dx_flat = outputs.copy()
        dx_flat[p.arange(outputs.shape[0]), targets] -= 1
        return dx_flat
