import numpy as np

from Layers.Core import Core


def Relu(Z):
    return np.maximum(.0, Z)


def ReluGrad(Z):
    res = np.zeros(Z.shape)
    res[Z > 0] = 1
    return res


def Sigmoid(Z):
    return 1./(1+np.exp(-Z))


def SigmoidGrad(Z):
    tmp = Sigmoid(Z)
    return tmp * (1-tmp)


def TanH(Z):
    return (np.exp(Z) - np.exp(-Z))/(np.exp(Z) + np.exp(-Z))


def TanHGrad(Z):
    tmp = TanH(Z)
    return 1-tmp**2


def Leak_Relu(Z):
    return np.maximum(0.01 * Z, Z)


def Leak_ReluGrad(Z):
    res = np.full(Z.shape, 0.01)
    res[Z > 0] = 1
    return res


def SoftmaxStep(Z):
    x = Z - np.max(Z, axis=1, keepdims=True)
    exp_x = np.exp(x)
    s = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return s


def SoftmaxGradStep(input=None):
    return np.ones(input.shape)


class Softmax(Core):
    def __index__(self, unitNumber):
        super().__init__(unitNumber, "Softmax")

    def forward(self, input):
        self.input = input
        return super().forward(input)

    def backward(self, dZ):
        return super().backward(dZ if dZ else self.input)


def get(Z, activation='relu'):
    if activation == 'sigmoid':
        A = Sigmoid(Z)
    elif activation == 'tanh':
        A = TanH(Z)
    elif activation == 'leak_relu':
        A = Leak_Relu(Z)
    elif activation == 'Softmax':
        A = SoftmaxStep(Z)
    elif activation == 'relu':
        A = Relu(Z)
    else:
        A = Z
    return A


def get_grad(dA, Z, activation='relu'):
    if activation == 'sigmoid':
        dZ = dA * SigmoidGrad(Z)
    elif activation == 'tanh':
        dZ = dA * TanHGrad(Z)
    elif activation == 'leak_relu':
        dZ = dA * Leak_ReluGrad(Z)
    elif activation == 'Softmax':
        dZ = dA * SoftmaxGradStep(Z)
    elif activation == 'relu':
        dZ = dA * ReluGrad(Z)
    else:
        dZ = dA
    return dZ
