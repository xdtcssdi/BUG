import Activation
from Layers.Layer import Layer
from Layers.Padding import *


class Convolution(Layer):

    def __init__(self, filter_count, filter_shape, stride=1, padding=0, activation='relu'):
        super().__init__(0, activation)
        self.filter_count = filter_count  # 卷积核数量
        self.filter_shape = filter_shape  # 卷积核形状
        self.stride = stride  # 步长
        self.padding = padding  # pad
        self.Z_pad = None
        print("Convolution Layer")

    def init_params(self, pre_nc):  # pre_nc 前一个通道数
        kernel_shape = (self.filter_shape[0], self.filter_shape[1], pre_nc, self.filter_count)
        self.W = np.random.randn(*kernel_shape)  # W.shape == (f, f ,pre_nc, nc)

        b_shape = [1, ]*len(self.W.shape)
        b_shape[-1] = self.filter_count
        self.b = np.random.randn(*tuple(b_shape))  # b.shape = (1, 1, 1, nc)

    def forward(self, input):
        self.input = input
        if not self.hasParams:
            self.init_params(input.shape[2])
            self.hasParams = True
        n_w = int((input.shape[0] + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
        n_h = int((input.shape[1] + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
        Z = np.zeros((n_w, n_h, self.filter_count, input.shape[-1]))

        Z_pad = ZeroPad(input, self.padding) if self.padding > 0 else input

        for i in range(input.shape[-1]):
            a_prev_pad = Z_pad[:, :, :, i]
            for w in range(n_w):
                for h in range(n_h):
                    for nc in range(self.filter_count):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]
                        a_slice = a_prev_pad[hs:he, vs:ve, :]
                        Z[w, h, nc, i] = np.sum(a_slice * self.W[:, :, :, nc] + self.b[:, :, :, nc])
        self.Z = Z
        self.A = Activation.get(Z, self.activation)
        return self.A

    def backward(self, dZ):
        dZ_pad = np.zeros_like(self.Z_pad)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        n_w, n_h, nc, m = dZ.shape

        for i in range(m):
            a_pre = self.Z_pad[:, :, :, i]
            da_pre = dZ_pad[:, :, :, i]
            for w in range(n_w):
                for h in range(n_h):
                    for c in range(nc):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]
                        a_slice = a_pre[hs:he, vs:ve, :]
                        da_pre[hs:he, vs:ve, :] += self.W[:, :, :, c] * dZ[w, h, c, i]
                        self.dW[:, :, :, c] += a_slice * dZ[w, h, c, i]
                        self.db[:, :, :, c] += dZ[w, h, c, i]
            self.dA[:, :, :, i] = da_pre[self.padding:-self.padding, self.padding:-self.padding, :]
        return Activation.get_grad(self.dA, self.Z, self.activation), self.dW, self.db

