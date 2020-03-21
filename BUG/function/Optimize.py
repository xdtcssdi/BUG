import os

from BUG.Layers.Layer import Pooling
from BUG.load_package import p
import numpy

class Optimize:
    def __init__(self, layers, theta=1e-2):
        self.layers = layers
        self.theta = theta

    def update(self, *arg):
        raise NotImplementedError

    def save_parameters(self, path):
        raise NotImplementedError

    def load_parameters(self, path):
        raise NotImplementedError

    def init_params(self, layers):
        raise NotImplementedError


class Momentum(Optimize):
    def __init__(self, layers, theta=1e-2):
        super(Momentum, self).__init__(layers, theta)
        self.name = 'Momentum'
        self.v = {}

    def init_params(self, layers):
        if self.v:
            return
        self.layers = layers
        for i in range(len(layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            for key in layer.parameters.keys():
                self.v['V_' + key + str(i)] = p.zeros_like(layer.parameters[key])
            if layer.batch_normal is not None:
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_' + key + str(i)] = p.zeros_like(layer.batch_normal.parameters[key])

    def save_parameters(self, path):
        p.savez_compressed(path + os.sep + self.name, **self.v)

    def load_parameters(self, path):
        v = p.load(path + os.sep + self.name + '.npz')
        if isinstance(v, numpy.lib.npyio.NpzFile):
            files = v.files
        else:
            files = v.npz_file.files

        for key in files:
            self.v[key] = v[key]

    def update(self, t, learning_rate, beta=0.9):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            gradients_clip(layer.gradients, self.theta)
            for key in layer.parameters.keys():
                self.v['V_' + key + str(i)] = beta * self.v['V_' + key + str(i)] + (1 - beta) * layer.gradients[key]
                layer.parameters[key] -= learning_rate * self.v['V_' + key + str(i)]
                del layer.gradients[key]
            if layer.batch_normal is not None:
                gradients_clip(layer.batch_normal.gradients, self.theta)
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_' + key + str(i)] = beta * self.v['V_' + key + str(i)] + (1 - beta) * \
                                                  layer.batch_normal.gradients[key]
                    layer.batch_normal.parameters[key] -= learning_rate * self.v['V_' + key + str(i)]
                    del layer.batch_normal.gradients[key]


class Adam(Optimize):
    def __init__(self, layers, theta=1e-2):
        super(Adam, self).__init__(layers, theta=1e-2)
        self.name = 'Adam'
        self.v = {}
        self.s = {}

    def init_params(self, layers):
        if self.v:
            return
        self.layers = layers
        for i in range(len(layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            for key in layer.parameters.keys():
                self.v['V_' + key + str(i)] = p.zeros_like(layer.parameters[key])
                self.s['S_' + key + str(i)] = p.zeros_like(layer.parameters[key])
            if layer.batch_normal is not None:
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_' + key + str(i)] = p.zeros_like(layer.batch_normal.parameters[key])
                    self.s['S_' + key + str(i)] = p.zeros_like(layer.batch_normal.parameters[key])

    def save_parameters(self, path):
        p.savez_compressed(path + os.sep + self.name + '_v', **self.v)
        p.savez_compressed(path + os.sep + self.name + '_s', **self.s)

    def load_parameters(self, path):
        v = p.load(path + os.sep + self.name + '_v.npz')
        if isinstance(v, numpy.lib.npyio.NpzFile):
            files = v.files
        else:
            files = v.npz_file.files

        for key in files:
            self.v[key] = v[key]
        s = p.load(path + os.sep + self.name + '_s.npz')

        if isinstance(s, numpy.lib.npyio.NpzFile):
            files = s.files
        else:
            files = s.npz_file.files
        for key in files:
            self.s[key] = s[key]

    def update(self, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue

            gradients_clip(layer.gradients, self.theta)
            for key in layer.parameters.keys():
                self.v['V_' + key + str(i)] = beta1 * self.v['V_' + key + str(i)] + (1 - beta1) * layer.gradients[key]
                self.s['S_' + key + str(i)] = beta2 * self.s['S_' + key + str(i)] + (1 - beta2) * p.square(
                    layer.gradients[key])
                layer.parameters[key] -= learning_rate * self.v['V_' + key + str(i)] / (1 - p.power(beta1, t)) / (
                        p.sqrt(self.s['S_' + key + str(i)] / (1 - p.power(beta2, t))) + epsilon)
                del layer.gradients[key]
            if layer.batch_normal is not None:
                gradients_clip(layer.batch_normal.gradients, self.theta)
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_' + key + str(i)] = beta1 * self.v['V_' + key + str(i)] + (1 - beta1) * \
                                                  layer.batch_normal.gradients[key]
                    self.s['S_' + key + str(i)] = beta2 * self.s['S_' + key + str(i)] + (1 - beta2) * p.square(
                        layer.batch_normal.gradients[key])
                    layer.batch_normal.parameters[key] -= learning_rate * self.v['V_' + key + str(i)] / (
                            1 - p.power(beta1, t)) / (p.sqrt(
                        self.s['S_' + key + str(i)] / (1 - p.power(beta2, t))) + epsilon)
                    del layer.batch_normal.gradients[key]


class BatchGradientDescent(Optimize):

    def __init__(self, layers, theta=1e-2):
        super(BatchGradientDescent, self).__init__(layers, theta)

    def save_parameters(self, path):
        pass

    def load_parameters(self, path):
        pass

    def init_params(self, layers):
        self.layers = layers

    def update(self, t, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Pooling):
                continue

            gradients_clip(layer.gradients, self.theta)
            for key in layer.parameters.keys():
                layer.parameters[key] -= learning_rate * layer.gradients[key]
                del layer.gradients[key]

            if layer.batch_normal is not None:
                gradients_clip(layer.batch_normal.gradients, self.theta)
                for key in layer.batch_normal.parameters.keys():
                    layer.batch_normal.parameters[key] -= learning_rate * layer.batch_normal.gradients[key]
                    del layer.batch_normal.gradients[key]


#  梯度裁剪
def gradients_clip(gradients, theta=5.):
    for key in gradients.keys():
        gradients[key] = theta / (p.linalg.norm(gradients[key]) + 1e-11) * gradients[key]
