from BUG.load_package import p


def Relu(Z):
    return p.maximum(.0, Z)


def ReluGrad(Z):
    res = p.zeros(Z.shape)
    res[Z > 0] = 1
    return res


def Sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = p.zeros_like(x)
    z[pos_mask] = p.exp(-x[pos_mask])
    z[neg_mask] = p.exp(x[neg_mask])
    top = p.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def SigmoidGrad(Z):
    return p.ones_like(Z)


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
    if Z.ndim == 3:
        N, T, D = Z.shape
        Z = Z.reshape(N * T, D)
    shift_scores = Z - p.max(Z, axis=1, keepdims=True)
    return p.exp(shift_scores) / p.sum(p.exp(shift_scores), axis=1, keepdims=True)


def SoftmaxGradStep(Z):
    return p.ones_like(Z)


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
    elif activation == 'linear':
        A = Z
    else:
        raise ValueError('激活方式错误')
    return A


def ac_get_grad(dA, Z, activation='relu'):
    if activation == 'sigmoid':
        dZ = dA * SigmoidGrad(Z)
    elif activation == 'tanh':
        dZ = dA * TanHGrad(Z)
    elif activation == 'leak_relu':
        dZ = dA * Leak_ReluGrad(Z)
    elif activation == 'softmax':
        dZ = dA * SoftmaxGradStep(Z)
    elif activation == 'relu':
        dZ = dA * ReluGrad(Z)
    elif activation == 'linear':
        dZ = dA
    else:
        raise ValueError('激活方式错误')
    return dZ


if __name__ == '__main__':
    print(SoftmaxStep(p.array([8.4022668, 6.88335424, 11.72859856, 28.16765532, 5.18627901, 12.12249185
                                  , 6.40666255, 10.16708038, 11.06757103, 7.41443078]).reshape(1, -1)).sum())
