from .Layer import Layer
from .Padding import ZeroPad
from .im2col import *


class PoolingForloop(Layer):
    def __init__(self, filter_shape, padding=0, stride=1, mode='max'):
        super(PoolingForloop, self).__init__()
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.mode = mode
        assert (self.mode in ['max', 'average'])

    def init_params(self, nx):
        pass

    def forward(self, A_pre, mode='train'):
        m, w, h, nc = A_pre.shape
        n_w = int((w + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
        n_h = int((h + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
        A = np.zeros((m, n_w, n_h, nc))

        self.A_pad = ZeroPad(A_pre, self.padding) if self.padding > 0 else A_pre

        if self.mode == 'max':
            for i in range(m):
                a_prev_pad = self.A_pad[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            a_slice = a_prev_pad[hs:he, vs:ve, c]
                            A[i, w, h, c] = np.max(a_slice)
        elif self.mode == 'average':
            for i in range(m):
                a_prev_pad = self.A_pad[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            a_slice = a_prev_pad[hs:he, vs:ve, c]
                            A[i, w, h, c] = np.mean(a_slice)
        return A

    def backward(self, dZ):
        m, n_w, n_h, nc = dZ.shape
        dA = np.zeros_like(self.A_pad)
        if self.mode == 'max':
            for i in range(m):
                da = dZ[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            dA[i, hs:he, vs:ve, c] += self.maxPooling_backward(self.A_pad[i, hs:he, vs:ve, c],
                                                                               da[w, h, c])
        elif self.mode == 'average':
            for i in range(m):
                da = dZ[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            dA[i, hs:he, vs:ve, c] += self.averagePooling_backward(da[w, h, c])

        return dA[:, self.padding:-self.padding, self.padding:-self.padding, :] if self.padding > 0 else dA

    def maxPooling_backward(self, z, grad):  # input: Z:matrix, grad is real return matrix
        assert (z.ndim == 2)
        return (z == np.max(z)) * grad

    def averagePooling_backward(self, a):  # input : a is real return matrix
        return np.ones((self.filter_shape[0], self.filter_shape[1])) * (
                a / (self.filter_shape[0] * self.filter_shape[1]))


class Pooling(Layer):
    def __init__(self, filter_shape, paddingMode='same', stride=1, mode='max'):
        super(Pooling, self).__init__()
        self.filter_shape = filter_shape
        self.name = 'Pooling'
        self.stride = stride
        self.padding = 0 if paddingMode == 'valid' else (filter_shape[0] - 1) // 2
        self.mode = mode
        assert (self.mode in ['max', 'average'])

    def init_params(self, nx):
        pass

    def forward(self, A_pre, mode='train'):
        N, C, H, W = A_pre.shape

        out_height = (H - self.filter_shape[0]) // self.stride + 1
        out_width = (W - self.filter_shape[1]) // self.stride + 1

        x_split = A_pre.reshape(N * C, 1, H, W)
        x_cols = im2col_indices(x_split, self.filter_shape[0], self.filter_shape[1], padding=self.padding,
                                stride=self.stride)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
        self.cache = (A_pre, x_cols, x_cols_argmax)
        return out

    def backward(self, dZ):
        x, x_cols, x_cols_argmax = self.cache
        del self.cache
        N, C, H, W = x.shape

        dout_reshaped = dZ.transpose(2, 3, 0, 1).flatten()
        dx_cols = np.zeros_like(x_cols)
        dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
        dx = col2im_indices(dx_cols, (N * C, 1, H, W), self.filter_shape[0], self.filter_shape[1],
                            padding=self.padding, stride=self.stride)
        dx = dx.reshape(x.shape)
        return dx
