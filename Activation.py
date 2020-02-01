import numpy as np


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

