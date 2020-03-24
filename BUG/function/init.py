import numpy
from BUG.load_package import p
import math


def orthogonal(shape):
    flat_shape = (shape[0], numpy.prod(shape[1:]))
    a = numpy.random.normal(0.0, 1.0, flat_shape)
    u, _, v = numpy.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return p.array(q.reshape(shape))


def he_normal(shape):
    if len(shape) > 2:
        in_dim = shape[1] * shape[2] * shape[3]
    else:
        in_dim, out_dim = shape
    return p.random.normal(.0, math.sqrt(2. / in_dim), shape)


def he_uniform(shape):
    if len(shape) > 2:
        in_dim = shape[1] * shape[2] * shape[3]
    else:
        in_dim, out_dim = shape
    return p.random.uniform(-math.sqrt(6. / in_dim), math.sqrt(6. / in_dim), shape)


def xavier_normal(shape):
    if len(shape) > 2:
        in_dim = shape[1] * shape[2] * shape[3]
        out_dim = shape[0] * shape[2] * shape[3]
    else:
        in_dim, out_dim = shape
    return p.random.normal(.0, math.sqrt(2. / (in_dim+out_dim)), shape)


def xavier_uniform(shape):
    if len(shape) > 2:
        in_dim = shape[1] * shape[2] * shape[3]
        out_dim = shape[0] * shape[2] * shape[3]
    else:
        in_dim, out_dim = shape
    t = math.sqrt(6. / (in_dim+out_dim))
    return p.random.uniform(-t, t, shape)


def normal(shape):
    return p.random.randn(*shape)


def get_init(activation, shape, option):
    if option == 'normal':
        if activation == 'relu' or activation == 'leak_relu':
            return he_normal(shape)
        else:
            return xavier_normal(shape)
    else:
        if activation == 'relu' or activation == 'leak_relu':
            return he_uniform(shape)
        else:
            return xavier_uniform(shape)

