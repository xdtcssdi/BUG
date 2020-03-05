from BUG.Layers.Layer import SimpleRNN, Pooling
from BUG.load_package import p


class Optimize:
    def __init__(self, layers):
        self.layers = layers

    def update(self, *arg):
        pass


class Momentum(Optimize):
    def __init__(self, layers):
        super(Momentum, self).__init__(layers)
        self.v = {}
        self.s = {}
        for i in range(len(layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            for key in layer.parameters.keys():
                self.v['V_d' + key + str(i)] = p.zeros_like(layer.parameters[key])
            if layer.batch_normal is not None:
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_d' + key + str(i)] = p.zeros_like(layer.batch_normal.parameters[key])

    def update(self, t, learning_rate, beta=0.9):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            gradients_clip(layer.gradients)
            for key in layer.parameters.keys():
                self.v['V_d' + key + str(i)] = beta * self.v['V_d' + key + str(i)] + (1 - beta) * layer.gradients[
                    'd' + key]
                layer.parameters[key] -= learning_rate * self.v['V_d' + key + str(i)]

            if layer.batch_normal is not None:
                gradients_clip(layer.batch_normal.gradients)
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_d' + key + str(i)] = beta * self.v['V_d' + key + str(i)] + (1 - beta) * \
                                                   layer.batch_normal.gradients['d' + key]
                    layer.batch_normal.parameters[key] -= learning_rate * self.v['V_d' + key + str(i)]


class Adam(Optimize):
    def __init__(self, layers):
        super(Adam, self).__init__(layers)
        self.v = {}
        self.s = {}
        for i in range(len(layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            for key in layer.parameters.keys():
                self.v['V_d' + key + str(i)] = p.zeros_like(layer.parameters[key])
                self.s['S_d' + key + str(i)] = p.zeros_like(layer.parameters[key])
            if layer.batch_normal is not None:
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_d' + key + str(i)] = p.zeros_like(layer.batch_normal.parameters[key])
                    self.s['S_d' + key + str(i)] = p.zeros_like(layer.batch_normal.parameters[key])

    def update(self, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue

            gradients_clip(layer.gradients)
            for key in layer.parameters.keys():
                self.v['V_d' + key + str(i)] = beta1 * self.v['V_d' + key + str(i)] + (1 - beta1) * layer.gradients['d' + key]
                self.s['S_d' + key + str(i)] = beta2 * self.s['S_d' + key + str(i)] + (1 - beta2) * p.square(layer.gradients['d' + key])
                layer.parameters[key] -= learning_rate * self.v['V_d' + key + str(i)] / (1 - p.power(beta1, t)) / (
                            p.sqrt(self.s['S_d' + key + str(i)] / (1 - p.power(beta2, t))) + epsilon)

            if layer.batch_normal is not None:
                gradients_clip(layer.batch_normal.gradients)
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_d' + key + str(i)] = beta1 * self.v['V_d' + key + str(i)] + (1 - beta1) * \
                                                   layer.batch_normal.gradients['d' + key]
                    self.s['S_d' + key + str(i)] = beta2 * self.s['S_d' + key + str(i)] + (1 - beta2) * p.square(
                        layer.batch_normal.gradients['d' + key])
                    layer.batch_normal.parameters[key] -= learning_rate * self.v['V_d' + key + str(i)] / (
                                1 - p.power(beta1, t)) / (p.sqrt(
                        self.s['S_d' + key + str(i)] / (1 - p.power(beta2, t))) + epsilon)


class BatchGradientDescent(Optimize):
    def __init__(self, layers):
        super(BatchGradientDescent, self).__init__(layers)

    def update(self, t, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Pooling):
                continue

            gradients_clip(layer.gradients)
            for key in layer.parameters.keys():
                layer.parameters[key] -= learning_rate * layer.gradients['d'+key]

            if layer.batch_normal is not None:
                gradients_clip(layer.batch_normal.gradients)
                for key in layer.batch_normal.parameters.keys():
                    layer.batch_normal.parameters[key] -= learning_rate * layer.batch_normal.gradients['d'+key]


#  梯度裁剪
def gradients_clip(gradients, theta=5.):
    for key in gradients.keys():
        gradients[key] = theta / (p.linalg.norm(gradients[key]) + 1e-11) * gradients[key]
