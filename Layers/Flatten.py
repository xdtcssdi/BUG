import numpy as np

import Activation
from Layers.Layer import Layer


class Flatten(Layer):

    def __init__(self, out_dims=2, activation='relu'):
        super(Flatten, self).__init__(0, activation)
        self.activation = activation
        self.out_dims = out_dims
        self.input_shape = None
        #print("Flatten layer")

    def init_params(self, nx):
        pass

    def forward(self, input):
        self.input_shape = input.shape
        A = Activation.get(input.reshape(-1, input.shape[-1]), self.activation)
        self.unit_number = A.shape[0]  # 展开后 (nx,m) 接全连接神经网络需要前一层的神经元数
        return A

    def backward(self, dZ):
        dA = np.dot(self.next_layer.W.T, dZ)
        return dA.reshape(self.input_shape)

