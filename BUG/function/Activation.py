from BUG.load_package import p

def Relu(Z):
    return p.maximum(.0, Z)


def ReluGrad(Z):
    res = p.zeros(Z.shape)
    res[Z > 0] = 1
    return res


def Sigmoid(Z):
    return .5 * (1 + p.tanh(.5 * Z))


def SigmoidGrad(Z):
    tmp = Sigmoid(Z)
    return tmp * (1 - tmp)


def TanH(Z):
    return p.tanh(Z)


def TanHGrad(Z):
    return 1 - p.tanh(Z) ** 2


def Leak_Relu(Z):
    return p.maximum(0.01 * Z, Z)


def Leak_ReluGrad(Z):
    res = p.full(Z.shape, 0.01)
    res[Z > 0] = 1
    return res


def SoftmaxStep(Z):
    shift_scores = Z - p.max(Z, axis=1).reshape(-1, 1) #  在每行中10个数都减去该行中最大的数字
    A = p.exp(shift_scores) / p.sum(p.exp(shift_scores), axis=1).reshape(-1, 1)
    return A


def SoftmaxGradStep(Z):
    idx = p.argmax(Z, axis=1)
    Z[:, idx] -= 1
    return Z


def ac_get(Z, activation='relu'):
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


def ac_get_grad(dA, Z, activation='relu'):
    if activation == 'sigmoid':
        dZ = dA * SigmoidGrad(Z)
    elif activation == 'tanh':
        dZ = dA * TanHGrad(Z)
    elif activation == 'leak_relu':
        dZ = dA * Leak_ReluGrad(Z)
    elif activation == 'softmax':
        dZ = SoftmaxGradStep(Z)
    elif activation == 'relu':
        dZ = dA * ReluGrad(Z)
    elif activation is None:
        dZ = dA
    else:
        raise ValueError
    return dZ
