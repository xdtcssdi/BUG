import math
import numpy as np

from function.Activation import Activation
from .Layer import Layer
from .Normalization import BatchNormal


class Core(Layer):

    def __init__(self, unit_number, activation="relu", batchNormal=False):
        super(Core, self).__init__(unit_number, activation)
        self.batchNormal = BatchNormal() if batchNormal else None
        self.name = 'Core'

    def forward(self, A_pre, mode='train'):
        self.A_pre = A_pre
        self.init_params(A_pre)
        self.Z = np.dot(A_pre, self.W) + self.b
        Zhat = self.batchNormal.forward(self.Z) if self.batchNormal else self.Z
        return Activation.get(Zhat, self.activation)

    def backward(self, dZ):
        dA = dZ if self.isLast else np.dot(dZ, self.next_layer.W.T)
        if self.batchNormal:
            dA = self.batchNormal.backward(dA)
        dZ = Activation.get_grad(dA, self.Z, self.activation)

        self.dW = np.divide(1., dZ.shape[0]) * np.dot(self.A_pre.T, dZ)
        self.db = np.mean(dZ, axis=0, keepdims=True)
        return dZ

    def init_params(self, A_pre):
        pre_unit = A_pre.shape[1] if self.isFirst else self.pre_layer.unit_number
        if self.W is None:
            if self.activation == 'relu' or self.activation == 'leak_relu':  # 'Xavier'
                self.W = np.random.uniform(-math.sqrt(6. / (pre_unit + self.unit_number)),
                                           math.sqrt(6. / (pre_unit + self.unit_number)),
                                           (pre_unit, self.unit_number))
            elif self.activation == 'tanh' or self.activation == 'sigmoid':
                self.W = np.random.uniform(-1., 1., (pre_unit, self.unit_number)) \
                         * np.sqrt(6./(pre_unit + self.unit_number))
            else:  # 'MSRA'
                self.W = np.random.normal(0, math.sqrt(2./pre_unit), size=(pre_unit, self.unit_number))
            # self.W = np.random.randn(pre_unit, self.unit_number) * 0.01
        if self.b is None:
            self.b = np.zeros((1, self.unit_number))

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db
