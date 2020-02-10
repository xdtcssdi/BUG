import sys
sys.path.append("../")
import Activation
from .Layer import Layer
from .Padding import *


class Convolution(Layer):

    def __init__(self, filter_count, filter_shape, stride=1, padding=0, activation='relu'):
        super(Convolution, self).__init__(0, activation)
        self.filter_count = filter_count  # 卷积核数量
        self.filter_shape = filter_shape  # 卷积核形状
        self.stride = stride  # 步长
        self.padding = padding  # pad
        self.Z_pad = None

    def init_params(self, pre_nc):  # pre_nc 前一个通道数
        kernel_shape = (self.filter_shape[0], self.filter_shape[1], pre_nc, self.filter_count)
        self.W = np.random.randn(*kernel_shape)  # W.shape == (f, f ,pre_nc, nc)

        b_shape = [1, ]*len(self.W.shape)
        b_shape[-1] = self.filter_count
        self.b = np.random.randn(*tuple(b_shape))  # b.shape = (1, 1, 1, nc)

    # 没问题
    def forward(self, input):
        self.input = input
        if not self.hasParams:
            self.init_params(input.shape[-1])
            self.hasParams = True
            n_w = int((input.shape[1] + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
            n_h = int((input.shape[2] + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
            self.Z = np.zeros((input.shape[0], n_w, n_h, self.filter_count))
            self.dW = np.zeros_like(self.W)
            self.db = np.zeros_like(self.b)
            self.dA = np.zeros_like(self.input)
            self.dZ_pad = np.zeros_like(self.Z)

        self.Z_pad = ZeroPad(input, self.padding) if self.padding > 0 else input

        for i in range(input.shape[0]):
            a_prev_pad = self.Z_pad[i]
            for w in range(self.Z.shape[1]):
                for h in range(self.Z.shape[2]):
                    for nc in range(self.filter_count):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]
                        a_slice = a_prev_pad[hs:he, vs:ve, :]
                        self.Z[i, w, h, nc] = np.sum(a_slice * self.W[:, :, :, nc] + self.b[:, :, :, nc])
        self.A = Activation.get(self.Z, self.activation)
        return self.A

    # 没问题
    def backward(self, dZ):
        dZ_pad = np.zeros_like(self.Z_pad)
        m, n_w, n_h, nc = dZ.shape

        if self.padding > 0:
            for i in range(m):
                a_pre = self.Z_pad[i]
                da_pre = dZ_pad[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            a_slice = a_pre[hs:he, vs:ve, :]
                            da_pre[hs:he, vs:ve, :] += self.W[:, :, :, c] * dZ[i, w, h, c]
                            self.dW[:, :, :, c] += a_slice * dZ[i, w, h, c]
                            self.db[:, :, :, c] += dZ[i, w, h, c]
                self.dA[i] = da_pre[self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            for i in range(m):
                a_pre = self.Z_pad[i]
                da_pre = dZ_pad[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            a_slice = a_pre[hs:he, vs:ve, :]
                            da_pre[hs:he, vs:ve, :] += self.W[:, :, :, c] * dZ[i, w, h, c]
                            self.dW[:, :, :, c] += a_slice * dZ[i, w, h, c]
                            self.db[:, :, :, c] += dZ[i, w, h, c]
                self.dA[i] = da_pre

        return Activation.get_grad(self.dA, self.input, self.activation)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db
