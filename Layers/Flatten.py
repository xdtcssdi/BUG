import numpy as np

import Activation
from Layers.Layer import Layer


class Flatten(Layer):

    def __init__(self, out_dims=2, activation='relu'):
        super().__init__(0, activation)
        self.activation = activation
        self.out_dims = out_dims
        self.input_shape = None
        print("Flatten layer")

    def init_params(self, nx):
        pass

    def forward(self, input):
        self.input_shape = input.shape
        shape = input.shape[:self.out_dims-1] + (-1, )
        return Activation.get(input.reshape(*shape), self.activation)

    def backward(self, dZ):
        return dZ.reshape(self.input_shape)

