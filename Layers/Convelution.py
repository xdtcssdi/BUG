import Activation
from Layers.Layer import Layer
import numpy as np

from Layers.Padding import *

#input : (w,h,nc,m)
class Convolution(Layer):

    def __init__(self, filter_count, filter_shape, stride=1, padding=0, activation = 'relu'):
        self.filter_count = filter_count  # 卷积核数量
        self.filter_shape = filter_shape  # 卷积核形状
        self.stride = stride
        self.padding = padding

    def init_params(self, pre_nc):  # pre_nc 前一个通道数
        w = list(self.filter_shape)
        w.append(pre_nc)
        w.append(self.filter_count)
        w = tuple(w)
        self.W = np.random.randn(*w)

        b_shape = [1,]*(len(self.W.shape)-1)
        b_shape.append(self.filter_count)
        shape_b =tuple(b_shape)
        self.b = np.random.randn(*shape_b)

    def forward(self, input):

        n_w = int((input.shape[1] + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
        n_h = int((input.shape[2] + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
        Z = np.zeros((input.shape[0], n_w, n_h, self.filter_count))

        Z_pad = ZeroPad(input, self.padding) if self.padding > 0 else input

        for i in range(input.shape[0]):
            a_prev_pad = Z_pad[i]
            for w in range(n_w):
                for h in range(n_h):
                    for nc in range(self.filter_count):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]
                        a_slice = a_prev_pad[hs:he, vs:ve, :]
                        Z[i, w, h, nc] = self.conv_step(a_slice, self.W[:, :, :, nc], self.b[:, :, :, nc])
        return Z

    def backward(self, pre_grad, pre_W):
        pass

    def conv_step(self, A, W, b):
        return np.sum(A * W + b)




if __name__ == '__main__':
    np.random.seed(1)
    X = np.random.randn(10, 4, 4, 3)
    a = Convolution(8, (2, 2), padding=2)
    a.init_params(3)
    Z = a.forward(X)
    print(np.mean(Z))

#input : (m,w,h,nc)
# class Convolution():
#
#     def __init__(self, filter_count, filter_shape, stride=1, padding=0, activation = 'relu'):
#         self.filter_count = filter_count  # 卷积核数量
#         self.filter_shape = filter_shape  # 卷积核形状
#         self.stride = stride
#         self.padding = padding
#
#     def init_params(self, pre_nc):  # pre_nc 前一个通道数
#         w = list(self.filter_shape)
#         w.append(pre_nc)
#         w.append(self.filter_count)
#         w = tuple(w)
#         self.W = np.random.randn(*w)
#
#         b_shape = [1,]*(len(self.W.shape)-1)
#         b_shape.append(self.filter_count)
#         shape_b =tuple(b_shape)
#         self.b = np.random.randn(*shape_b)
#
#     def forward(self, input):
#
#         n_w = int((input.shape[0] + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
#         n_h = int((input.shape[1] + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
#         Z = np.zeros((n_w, n_h, self.filter_count, input.shape[-1]))
#
#         Z_pad = ZeroPad(input, self.padding) if self.padding > 0 else input
#
#         for i in range(input.shape[-1]):
#             a_prev_pad = Z_pad[:, :, :, i]
#             for w in range(n_w):
#                 for h in range(n_h):
#                     for nc in range(self.filter_count):
#                         hs = w * self.stride
#                         he = hs + self.filter_shape[0]
#                         vs = h * self.stride
#                         ve = vs + self.filter_shape[1]
#                         a_slice = a_prev_pad[hs:he, vs:ve, :]
#                         Z[w, h, nc, i] = self.conv_step(a_slice, self.W[:, :, :, nc], self.b[:, :, :, nc])
#         return Z
#
#     def backward(self, pre_grad, pre_W):
#         pass
#
#     def conv_step(self, A, W, b):
#         return np.sum(A * W + b)
#
# def ZeroPad(Z, pad = 0):
#     Z_pad = np.pad(Z, ((pad, pad), (pad, pad), (0, 0),  (0, 0), ), 'constant')
#     return Z_pad
#
#
# if __name__ == '__main__':
#     np.random.seed(1)
#     X = np.random.randn(4, 4, 3, 10)
#     a = Convolution(8, (2, 2), padding=2)
#     a.init_params(3)
#     Z = a.forward(X)
#     print(np.mean(Z))
