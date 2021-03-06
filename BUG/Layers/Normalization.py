from BUG.load_package import p


class BatchNormal:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self.parameters = {}
        self.gradients = {}
        self.caches = None
        self.running_mean = None
        self.running_var = None
        self.sqrtvar = None

    def save_params(self, filename):
        p.savez_compressed(filename, epsilon=self.epsilon, gamma=self.parameters['gamma'],
                           beta=self.parameters['beta'], running_mean=self.running_mean, sqrtvar=self.sqrtvar,
                           running_var=self.running_var)

    def load_params(self, filename):
        r = p.load(filename)
        self.epsilon = r['epsilon']
        self.parameters['gamma'] = r['gamma']
        self.parameters['beta'] = r['beta']
        self.running_mean = r['running_mean']
        self.running_var = r['running_var']
        self.sqrtvar = r['sqrtvar']

    def init_params(self, nx):
        pass

    def forward(self, A_pre, mode='train', momentum=0.9):
        if A_pre.ndim == 4:
            return self.fourDims_batchnorm_forward(A_pre, mode, momentum)
        elif A_pre.ndim == 2:
            return self.twoDims_batchnormal_forward(A_pre, mode, momentum)
        elif A_pre.ndim == 3:
            return self.threeDims_batchnorm_forward(A_pre, mode, momentum)
        else:
            raise ValueError

    def backward(self, pre_grad):
        if pre_grad.ndim == 4:
            return self.fourDims_batchnorm_backward(pre_grad)
        elif pre_grad.ndim == 2:
            return self.twoDims_batchnormal_backward(pre_grad)
        elif pre_grad.ndim == 3:
            return self.threeDims_batchnorm_backward(pre_grad)
        else:
            raise ValueError

    def fourDims_batchnorm_forward(self, A_pre, mode='train', momentum=0.9):
        N, C, H, W = A_pre.shape
        x_flat = A_pre.reshape(N * H * W, C)
        out_flat = self.twoDims_batchnormal_forward(x_flat, mode, momentum)
        out = out_flat.reshape(A_pre.shape)
        return out

    def fourDims_batchnorm_backward(self, dout):
        N, C, H, W = dout.shape
        dout_flat = dout.reshape(N * H * W, C)
        dx_flat = self.twoDims_batchnormal_backward(dout_flat)
        dx = dx_flat.reshape(dout.shape)
        return dx

    def threeDims_batchnorm_forward(self, A_pre, mode='train', momentum=0.9):
        N, T, D = A_pre.shape
        x_flat = A_pre.reshape(N * T, D)
        out_flat = self.twoDims_batchnormal_forward(x_flat, mode, momentum)
        out = out_flat.reshape(A_pre.shape)
        return out

    def threeDims_batchnorm_backward(self, dout):
        N, T, D = dout.shape
        dout_flat = dout.reshape(N * T, D)
        dx_flat = self.twoDims_batchnormal_backward(dout_flat)
        dx = dx_flat.reshape(dout.shape)
        return dx

    def twoDims_batchnormal_forward(self, A_pre, mode='train', momentum=0.9):
        if mode == 'train':
            if 'beta' not in self.parameters:
                self.parameters['beta'] = p.zeros((A_pre.shape[-1],))
                self.parameters['gamma'] = p.ones_like(self.parameters['beta'])
            mean = p.mean(A_pre, axis=0)
            xmu = A_pre - mean
            var = p.mean(xmu ** 2, axis=0)

            if self.running_mean is None:
                self.running_mean = p.zeros(A_pre.shape[-1], dtype=A_pre.dtype)
                self.running_var = p.zeros(A_pre.shape[-1], dtype=A_pre.dtype)
            self.running_mean = momentum * self.running_mean + (1 - momentum) * mean
            self.running_var = momentum * self.running_var + (1 - momentum) * var

            self.sqrtvar = p.sqrt(var + self.epsilon)
            ivar = 1. / self.sqrtvar
            xhat = xmu * ivar
            gammax = self.parameters['gamma'] * xhat
            out = gammax + self.parameters['beta']
            self.caches = (xhat, xmu, ivar, self.sqrtvar)
        elif mode == 'test':
            scale = self.parameters['gamma'] / self.sqrtvar
            out = A_pre * scale + (self.parameters['beta'] - self.running_mean * scale)
        else:
            raise ValueError
        return out

    def twoDims_batchnormal_backward(self, pre_grad):
        xhat, xmu, ivar, sqrtvar = self.caches
        del self.caches
        m, nx = pre_grad.shape
        self.gradients['beta'] = p.sum(pre_grad, axis=0)
        dgammax = pre_grad
        self.gradients['gamma'] = p.sum(xhat * dgammax, axis=0)
        dxhat = self.parameters['gamma'] * dgammax
        divar = p.sum(xmu * dxhat, axis=0)
        dxmu1 = dxhat * ivar
        dsqrtvar = -1. / (sqrtvar ** 2) * divar
        dvar = 0.5 * ivar * dsqrtvar
        dsq = p.divide(1., m) * p.ones_like(pre_grad) * dvar
        dxmu2 = 2 * dsq * xmu
        dx1 = dxmu1 + dxmu2
        dmu = -1 * p.sum(dx1, axis=0)
        dx2 = p.divide(1., m) * p.ones_like(pre_grad) * dmu
        dx = dx1 + dx2
        return dx
