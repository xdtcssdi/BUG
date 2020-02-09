from Layers.Flatten import Flatten
from Layers.Layer import Layer
import numpy as np
import Activation


class Core(Layer):

    def __init__(self, unit_number,  activation="relu"):
        super(Core, self).__init__(unit_number, activation)
        #print("Core Layer")

    def forward(self, input):
        self.input = input
        if not self.hasParams:
            self.init_params(input)
            self.hasParams = True
        self.Z = np.dot(self.W, input) + self.b
        self.A = Activation.get(self.Z, self.activation)
        return self.A

    def backward(self, dZ):
        if self.isLast:
            self.dA = dZ
        else:
            self.dA = np.dot(self.next_layer.W.T, dZ)
        self.dZ = Activation.get_grad(self.dA, self.Z, self.activation)
        self.dW = 1. / dZ.shape[-1] * np.dot(self.dZ, self.input.T)
        self.db = np.mean(self.dZ, axis=1, keepdims=True)
        return self.dZ

    def init_params(self, input):
        pre_unit = input.shape[0] if self.isFirst else self.pre_layer.unit_number
        self.W = np.random.randn(self.unit_number, pre_unit) / np.sqrt(pre_unit)
        self.b = np.zeros((self.unit_number, 1))

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db