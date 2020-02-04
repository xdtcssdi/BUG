
import Activation
from Layers.Layer import Layer
import numpy as np

class Core(Layer):

    def __init__(self, unit_number,  activation="relu"):
        super().__init__(unit_number, activation)

    def forward(self, input):
        self.input = input

        self.Z = np.dot(self.W, input) + self.b
        A = Activation.get(self.Z, self.activation)
        return A

    def backward(self, pre_grad, pre_W = None, last=False):
        if last:
            self.dA = pre_grad
        else:
            self.dA = np.dot(pre_W.T, pre_grad)

        self.dZ = Activation.get_grad(self.dA, self.Z, self.activation)

        self.dW = 1. / pre_grad.shape[-1] * np.dot(self.dZ, self.input.T)
        self.db = np.mean(self.dZ, axis=1, keepdims=True)
        return self.dZ, self.dW, self.db

    def init_params(self,nx):
        self.W = np.random.randn(self.unit_number, nx) * 0.01
        self.b = np.zeros((self.unit_number, 1))

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db