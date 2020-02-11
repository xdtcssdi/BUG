import numpy as np


class BatchNormal:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.dbeta = None
        self.dgamma = None
        self.caches = None
        self.running_mean = None
        self.running_var = None

    def init_params(self, nx):
        pass

    def forward(self, input, mode='train', momentum=0.9):
        if len(input.shape) == 4:
            return self.fourDims_batchnorm_forward(input, mode, momentum)
        elif len(input.shape) == 2:
            return self.twoDims_batchnormal_forward(input, mode, momentum)
        else:
            raise ValueError

    def backward(self, pre_grad):
        if len(pre_grad.shape) == 4:
            return self.fourDims_batchnorm_backward(pre_grad)
        elif len(pre_grad.shape) == 2:
            return self.twoDims_batchnormal_backward(pre_grad)
        else:
            raise ValueError

    @property
    def params(self):
        return self.gamma, self.beta

    @property
    def grads(self):
        return self.dgamma, self.dbeta

    def fourDims_batchnorm_forward(self, x, mode='train', momentum=0.9):
        N, W, H, C = x.shape
        x_flat = x.reshape(-1, C)
        out_flat = self.twoDims_batchnormal_forward(x_flat, mode, momentum)
        out = out_flat.reshape(x.shape)
        return out

    def fourDims_batchnorm_backward(self, dout):
        N, W, H, C = dout.shape
        dout_flat = dout.reshape(-1, C)
        dx_flat = self.twoDims_batchnormal_backward(dout_flat)
        dx = dx_flat.reshape(dout.shape)
        return dx

    def twoDims_batchnormal_forward(self, input, mode='train', momentum=0.9):
        if mode == 'train':
            if self.beta is None:
                self.beta = np.zeros((input.shape[-1],))
                self.gamma = np.ones_like(self.beta)
            mean = np.mean(input, axis=0)
            xmu = input - mean
            var = np.mean(xmu ** 2, axis=0)

            if self.running_mean is None:
                self.running_mean = np.zeros(input.shape[-1], dtype=input.dtype)
                self.running_var = np.zeros(input.shape[-1], dtype=input.dtype)
            self.running_mean = momentum * self.running_mean + (1 - momentum) * mean
            self.running_var = momentum * self.running_var + (1 - momentum) * var

            self.sqrtvar = np.sqrt(var + self.epsilon)
            ivar = 1. / self.sqrtvar
            xhat = xmu * ivar
            gammax = self.gamma * xhat
            out = gammax + self.beta
            self.caches = (xhat, xmu, ivar, self.sqrtvar)
        elif mode == 'test':
            scale = self.gamma / self.sqrtvar
            out = input * scale + (self.beta - self.running_mean * scale)
        else:
            raise ValueError
        return out

    def twoDims_batchnormal_backward(self, pre_grad):
        xhat, xmu, ivar, sqrtvar = self.caches
        m, nx = pre_grad.shape
        self.dbeta = np.sum(pre_grad, axis=0)
        dgammax = pre_grad
        self.dgamma = np.sum(xhat * dgammax, axis=0)
        dxhat = self.gamma * dgammax
        divar = np.sum(xmu * dxhat, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. / (sqrtvar ** 2) * divar
        dvar = 0.5 * ivar * dsqrtvar
        dsq = 1. / m * np.ones_like(pre_grad) * dvar
        dxmu2 = 2 * dsq * xmu
        dx1 = dxmu1 + dxmu2
        dmu = -1 * np.sum(dx1, axis=0)
        dx2 = 1. / m * np.ones_like(pre_grad) * dmu
        dx = dx1 + dx2
        return dx
