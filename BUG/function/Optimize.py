from BUG.Layers.Layer import SimpleRNN, Pooling
from BUG.load_package import p


class Optimize:
    def __init__(self, layers, theta):
        self.layers = layers
        self.theta = theta

    def update(self, *arg):
        pass


class Momentum(Optimize):
    def __init__(self, layers,theta):
        super(Momentum, self).__init__(layers, theta)
        self.v = {}
        self.s = {}
        for i in range(len(layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            for key in layer.parameters.keys():
                self.v['V_' + key + str(i)] = p.zeros_like(layer.parameters[key])
            if layer.batch_normal is not None:
                for key in layer.batch_normal.parameters.keys():
                    self.v['V_' + key + str(i)] = p.zeros_like(layer.batch_normal.parameters[key])

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
    def __init__(self, layers, theta):
        super(Adam, self).__init__(layers, theta)
        self.v = {}
        self.s = {}
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

    def update(self, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue

            gradients_clip(layer.gradients, self.theta)
            for key in layer.parameters.keys():
                self.v['V_' + key + str(i)] = beta1 * self.v['V_' + key + str(i)] + (1 - beta1) * layer.gradients[key]
                self.s['S_' + key + str(i)] = beta2 * self.s['S_' + key + str(i)] + (1 - beta2) * p.square(layer.gradients[key])
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
    def __init__(self, layers, theta):
        super(BatchGradientDescent, self).__init__(layers, theta)

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
