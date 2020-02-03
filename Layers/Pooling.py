from Layers.Layer import Layer
import numpy as np

from Layers.Padding import ZeroPad


class Pooling(Layer):
    def __init__(self, filter_shape, padding=0, stride=1, mode='max'):
        super().__init__(0)
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.mode = mode
        assert (self.mode in ['max', 'average'])

    def init_params(self, nx):
        pass

    def forward(self, input):
        self.input = input
        m, w, h, nc = input.shape
        n_w = int((w + 2*self.padding - self.filter_shape[0]) / self.stride + 1)
        n_h = int((h + 2*self.padding - self.filter_shape[1]) / self.stride + 1)
        self.A_shape = (m, n_w, n_h, nc)
        A = np.zeros(self.A_shape)

        self.A_pad = ZeroPad(input, self.padding) if self.padding > 0 else input

        for i in range(input.shape[0]):
            a_prev_pad = self.A_pad[i]
            for w in range(n_w):
                for h in range(n_h):
                    for c in range(nc):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]
                        a_slice = a_prev_pad[hs:he, vs:ve, c]

                        if self.mode == 'max':
                            A[i, w, h, c] = np.max(a_slice)
                        elif self.mode == 'average':
                            A[i, w, h, c] = np.mean(a_slice)
        return A

    def backward(self, pre_grad, pre_W=None):
        m, n_w, n_h, nc = self.A_shape
        dA = np.zeros_like(self.A_pad)
        for i in range(m):
            da = pre_grad[i]
            for w in range(n_w):
                for h in range(n_h):
                    for c in range(nc):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]

                        if self.mode == 'max':
                            dA[i, hs:he, vs:ve, c] += self.maxPooling_backward(self.A_pad[i, hs:he, vs:ve, c],
                                                                               da[hs, vs, c])
                        elif self.mode == 'average':
                            dA[i, hs:he, vs:ve, c] += self.averagePooling_backward(da[hs, vs, c])
        return dA[:, self.padding:-self.padding, self.padding:-self.padding, :] if self.padding > 0 else dA

    def maxPooling_backward(self, Z, grad):  # input: Z:matrix, grad is real return matrix
        assert (len(Z.shape) == 2)
        return (Z == np.max(Z)) * grad

    def averagePooling_backward(self, a):  # input : a is real return matrix
        return np.ones((self.filter_shape[0], self.filter_shape[1])) * (a/(self.filter_shape[0]*self.filter_shape[1]))
