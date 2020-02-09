from Layers.Layer import Layer
import numpy as np

from Layers.Padding import ZeroPad


class Pooling(Layer):
    def __init__(self, filter_shape, padding=0, stride=1, mode='max'):
        super(Pooling, self).__init__(0)
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.mode = mode
        #print("Pooling")
        assert (self.mode in ['max', 'average'])

    def init_params(self, nx):
        pass

    def forward(self, input):
        #print("Pooling前", input.shape)
        self.input = input
        w, h, nc, m = input.shape
        n_w = int((w + 2*self.padding - self.filter_shape[0]) / self.stride + 1)
        n_h = int((h + 2*self.padding - self.filter_shape[1]) / self.stride + 1)
        self.A_shape = (n_w, n_h, nc, m)
        A = np.zeros(self.A_shape)

        self.A_pad = ZeroPad(input, self.padding) if self.padding > 0 else input

        for i in range(input.shape[-1]):
            a_prev_pad = self.A_pad[:, :, :, i]
            for w in range(n_w):
                for h in range(n_h):
                    for c in range(nc):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]
                        a_slice = a_prev_pad[hs:he, vs:ve, c]

                        if self.mode == 'max':
                            A[w, h, c, i] = np.max(a_slice)
                        elif self.mode == 'average':
                            A[w, h, c, i] = np.mean(a_slice)
        #print("Pooling后", A.shape)
        return A

    def backward(self, dZ):
        n_w, n_h, nc, m = dZ.shape
        dA = np.zeros_like(self.A_pad)
        for i in range(m):
            da = dZ[:, :, :, i]
            for w in range(n_w):
                for h in range(n_h):
                    for c in range(nc):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]
                        if self.mode == 'max':
                            dA[hs:he, vs:ve, c, i] += self.maxPooling_backward(self.A_pad[hs:he, vs:ve, c, i],
                                                                               da[w, h, c])
                        elif self.mode == 'average':
                            dA[hs:he, vs:ve, c, i] += self.averagePooling_backward(da[hs, vs, c])
        return dA[self.padding:-self.padding, self.padding:-self.padding, :, :] if self.padding > 0 else dA

    def maxPooling_backward(self, z, grad):  # input: Z:matrix, grad is real return matrix
        assert (len(z.shape) == 2)
        return (z == np.max(z)) * grad

    def averagePooling_backward(self, a):  # input : a is real return matrix
        return np.ones((self.filter_shape[0], self.filter_shape[1])) * (a/(self.filter_shape[0]*self.filter_shape[1]))
