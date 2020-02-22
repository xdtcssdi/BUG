import numpy as np

from .Activation import *
from .Layer import Layer


class Flatten(Layer):

    def __init__(self, out_dims=2, activation=None):
        super(Flatten, self).__init__(activation=activation)
        self.out_dims = out_dims
        self.input_shape = None
        self.name = 'Flatten'

    def init_params(self, nx):
        pass

    def forward(self, A_pre, mode='train'):  # m,1,28,28
        self.input_shape = A_pre.shape
        A = get(A_pre.reshape(A_pre.shape[0], -1), self.activation)
        self.unit_number = A.shape[-1]  # 展开后 (m, nx) 接全连接神经网络需要前一层的神经元数
        return A

    def backward(self, dZ):
        dA = np.dot(dZ, self.next_layer.W.T)
        return dA.reshape(self.input_shape)
