import numpy as np
np.seterr(divide='ignore',invalid='ignore')


def Relu(Z):
    return np.maximum(.0, Z)


def ReluGrad(Z):
    res = np.zeros(Z.shape)
    res[Z > 0] = 1
    return res


def Sigmoid(Z):
    return .5 * (1 + np.tanh(.5 * Z))


def SigmoidGrad(Z):
    tmp = Sigmoid(Z)
    return tmp * (1 - tmp)


def TanH(Z):
    return np.tanh(Z)


def TanHGrad(Z):
    return 1 - TanH(Z) ** 2


def Leak_Relu(Z):
    return np.maximum(0.01 * Z, Z)


def Leak_ReluGrad(Z):
    res = np.full(Z.shape, 0.01)
    res[Z > 0] = 1
    return res


def SoftmaxStep(Z):
    shift_scores = Z - np.max(Z, axis=1).reshape(-1, 1)                    #在每行中10个数都减去该行中最大的数字
    A = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
    return A


def SoftmaxGradStep(Z):
    N = Z.shape[0]
    dscores = SoftmaxStep(Z)
    dscores[range(N), list(Z)] -= 1
    dscores /= N
    return dscores


class Activation:

    def get(Z, activation='relu'):
        if activation == 'sigmoid':
            A = Sigmoid(Z)
        elif activation == 'tanh':
            A = TanH(Z)
        elif activation == 'leak_relu':
            A = Leak_Relu(Z)
        elif activation == 'softmax':
            A = SoftmaxStep(Z)
        elif activation == 'relu':
            A = Relu(Z)
        elif activation is None:
            A = Z
        else:
            raise ValueError
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
