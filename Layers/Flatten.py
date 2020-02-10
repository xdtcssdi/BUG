import numpy as np
from .Layer import Layer
import Activation

class Flatten(Layer):

    def __init__(self, out_dims=2, activation='None'):
        super(Flatten, self).__init__(0, activation)
        self.out_dims = out_dims
        self.input_shape = None

    def init_params(self, nx):
        pass

    def forward(self, input):  # m,28,28,1
        self.input_shape = input.shape
        A = Activation.get(input.reshape(input.shape[0], -1), self.activation)
        self.unit_number = A.shape[-1]  # 展开后 (m, nx) 接全连接神经网络需要前一层的神经元数
        return A

    def backward(self, dZ):
        self.dA = np.dot(dZ, self.next_layer.W.T)
        self.dZ = self.dA.reshape(self.input_shape)
        return self.dZ

