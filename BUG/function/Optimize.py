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
            if isinstance(layer, SimpleRNN):
                self.v['V_dWaa' + str(i)] = p.zeros(layer.Waa.shape)
                self.v['V_dWax' + str(i)] = p.zeros(layer.Wax.shape)
                self.v['V_dWya' + str(i)] = p.zeros(layer.Wya.shape)
                self.v['V_db' + str(i)] = p.zeros(layer.b.shape)
                self.v['V_dby' + str(i)] = p.zeros(layer.by.shape)
            else:
                self.v['V_dW' + str(i)] = p.zeros(layer.W.shape)
                self.v['V_db' + str(i)] = p.zeros(layer.b.shape)
            if layer.batch_normal is not None:
                self.v['V_dbeta' + str(i)] = p.zeros(layer.batch_normal.dbeta.shape)
                self.v['V_dgamma' + str(i)] = p.zeros(layer.batch_normal.dgamma.shape)

    def update(self, t, learning_rate, beta=0.9):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            if isinstance(layer, SimpleRNN):
                layer.dWaa, layer.dWax, layer.dWya, layer.db, layer.dby = gradients_clip(layer.dWaa, layer.dWax,
                                                                                         layer.dWya, layer.db,
                                                                                         layer.dby)
                self.v['V_dWaa' + str(i)] = beta * self.v['V_dWaa' + str(i)] + (1 - beta) * layer.dWaa
                self.v['V_dWax' + str(i)] = beta * self.v['V_dWax' + str(i)] + (1 - beta) * layer.dWax
                self.v['V_dWya' + str(i)] = beta * self.v['V_dWya' + str(i)] + (1 - beta) * layer.dWya
                self.v['V_db' + str(i)] = beta * self.v['V_db' + str(i)] + (1 - beta) * layer.db
                self.v['V_dby' + str(i)] = beta * self.v['V_dby' + str(i)] + (1 - beta) * layer.dby
                layer.Waa -= learning_rate * self.v['V_dWaa' + str(i)]
                layer.Wax -= learning_rate * self.v['V_dWax' + str(i)]
                layer.Wya -= learning_rate * self.v['V_dWya' + str(i)]
                layer.b -= learning_rate * self.v['V_db' + str(i)]
                layer.by -= learning_rate * self.v['V_dby' + str(i)]
                del layer.dWaa, layer.dWax, layer.dWya, layer.db, layer.dby

            else:
                layer.dW, layer.db = gradients_clip(layer.dW, layer.db)
                self.v['V_dW' + str(i)] = beta * self.v['V_dW' + str(i)] + (1 - beta) * layer.dW
                self.v['V_db' + str(i)] = beta * self.v['V_db' + str(i)] + (1 - beta) * layer.db
                layer.W -= learning_rate * self.v['V_dW' + str(i)]
                layer.b -= learning_rate * self.v['V_db' + str(i)]
                del layer.dW, layer.db

            if layer.batch_normal is not None:
                layer.batch_normal.dbeta, layer.batch_normal.dgamma = gradients_clip(layer.batch_normal.dbeta,
                                                                                     layer.batch_normal.dgamma)
                self.v['V_dbeta' + str(i)] = beta * self.v['V_dbeta' + str(i)] + (
                        1 - beta) * layer.batch_normal.dbeta
                self.v['V_dgamma' + str(i)] = beta * self.v['V_dgamma' + str(i)] + (
                        1 - beta) * layer.batch_normal.dgamma
                layer.batch_normal.beta -= learning_rate * self.v['V_dbeta' + str(i)]
                layer.batch_normal.gamma -= learning_rate * self.v['V_dgamma' + str(i)]
                del layer.batch_normal.dbeta, layer.batch_normal.dgamma


class Adam(Optimize):
    def __init__(self, layers):
        super(Adam, self).__init__(layers)
        self.v = {}
        self.s = {}
        for i in range(len(layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            if isinstance(layer, SimpleRNN):
                self.v['V_dWaa' + str(i)] = p.zeros(layer.Waa.shape)
                self.v['V_dWax' + str(i)] = p.zeros(layer.Wax.shape)
                self.v['V_dWya' + str(i)] = p.zeros(layer.Wya.shape)
                self.v['V_db' + str(i)] = p.zeros(layer.b.shape)
                self.v['V_dby' + str(i)] = p.zeros(layer.by.shape)
                self.s['S_dWaa' + str(i)] = p.zeros(layer.Waa.shape)
                self.s['S_dWax' + str(i)] = p.zeros(layer.Wax.shape)
                self.s['S_dWya' + str(i)] = p.zeros(layer.Wya.shape)
                self.s['S_db' + str(i)] = p.zeros(layer.b.shape)
                self.s['S_dby' + str(i)] = p.zeros(layer.by.shape)
            else:
                self.v['V_dW' + str(i)] = p.zeros(layer.W.shape)
                self.v['V_db' + str(i)] = p.zeros(layer.b.shape)
                self.s['S_dW' + str(i)] = p.zeros(layer.W.shape)
                self.s['S_db' + str(i)] = p.zeros(layer.b.shape)
            if layer.batch_normal is not None:
                self.v['V_dbeta' + str(i)] = p.zeros(layer.batch_normal.beta.shape)
                self.v['V_dgamma' + str(i)] = p.zeros(layer.batch_normal.gamma.shape)
                self.s['S_dbeta' + str(i)] = p.zeros(layer.batch_normal.beta.shape)
                self.s['S_dgamma' + str(i)] = p.zeros(layer.batch_normal.gamma.shape)

    def update(self, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Pooling):
                continue
            if isinstance(layer, SimpleRNN):
                layer.dWaa, layer.dWax, layer.dWya, layer.db, layer.dby = gradients_clip(layer.dWaa, layer.dWax,
                                                                                         layer.dWya, layer.db,
                                                                                         layer.dby)
                self.v['V_dWaa' + str(i)] = beta1 * self.v['V_dWaa' + str(i)] + (1 - beta1) * layer.dWaa
                self.v['V_dWax' + str(i)] = beta1 * self.v['V_dWax' + str(i)] + (1 - beta1) * layer.dWax
                self.v['V_dWya' + str(i)] = beta1 * self.v['V_dWya' + str(i)] + (1 - beta1) * layer.dWya
                self.v['V_db' + str(i)] = beta1 * self.v['V_db' + str(i)] + (1 - beta1) * layer.db
                self.v['V_dby' + str(i)] = beta1 * self.v['V_dby' + str(i)] + (1 - beta1) * layer.dby

                self.s['S_dWaa' + str(i)] = beta2 * self.s['S_dWaa' + str(i)] + (1 - beta2) * p.square(layer.dWaa)
                self.s['S_dWax' + str(i)] = beta2 * self.s['S_dWax' + str(i)] + (1 - beta2) * p.square(layer.dWax)
                self.s['S_dWya' + str(i)] = beta2 * self.s['S_dWya' + str(i)] + (1 - beta2) * p.square(layer.dWya)
                self.s['S_db' + str(i)] = beta2 * self.s['S_db' + str(i)] + (1 - beta2) * p.square(layer.db)
                self.s['S_dby' + str(i)] = beta2 * self.s['S_dby' + str(i)] + (1 - beta2) * p.square(layer.dby)
                V_dwaa_corrected = self.v['V_dWaa' + str(i)] / (1 - p.power(beta1, t))
                V_dwax_corrected = self.v['V_dWax' + str(i)] / (1 - p.power(beta1, t))
                V_dwya_corrected = self.v['V_dWya' + str(i)] / (1 - p.power(beta1, t))
                V_db_corrected = self.v['V_db' + str(i)] / (1 - p.power(beta1, t))
                V_dby_corrected = self.v['V_dby' + str(i)] / (1 - p.power(beta1, t))

                S_dwaa_corrected = self.s['S_dWaa' + str(i)] / (1 - p.power(beta2, t))
                S_dwax_corrected = self.s['S_dWax' + str(i)] / (1 - p.power(beta2, t))
                S_dwya_corrected = self.s['S_dWya' + str(i)] / (1 - p.power(beta2, t))
                S_db_corrected = self.s['S_db' + str(i)] / (1 - p.power(beta2, t))
                S_dby_corrected = self.s['S_dby' + str(i)] / (1 - p.power(beta2, t))

                layer.Waa -= learning_rate * V_dwaa_corrected / (p.sqrt(S_dwaa_corrected) + epsilon)
                layer.Wax -= learning_rate * V_dwax_corrected / (p.sqrt(S_dwax_corrected) + epsilon)
                layer.Wya -= learning_rate * V_dwya_corrected / (p.sqrt(S_dwya_corrected) + epsilon)
                layer.by -= learning_rate * V_dby_corrected / (p.sqrt(S_dby_corrected) + epsilon)
                layer.b -= learning_rate * V_db_corrected / (p.sqrt(S_db_corrected) + epsilon)

                del layer.dWaa, layer.dWax, layer.dWya, layer.db, layer.dby
            else:
                layer.dW, layer.db = gradients_clip(layer.dW, layer.db)
                self.v['V_dW' + str(i)] = beta1 * self.v['V_dW' + str(i)] + (1 - beta1) * layer.dW
                self.v['V_db' + str(i)] = beta1 * self.v['V_db' + str(i)] + (1 - beta1) * layer.db
                self.s['S_dW' + str(i)] = beta2 * self.s['S_dW' + str(i)] + (1 - beta2) * p.square(layer.dW)
                self.s['S_db' + str(i)] = beta2 * self.s['S_db' + str(i)] + (1 - beta2) * p.square(layer.db)
                V_dw_corrected = self.v['V_dW' + str(i)] / (1 - p.power(beta1, t))
                V_db_corrected = self.v['V_db' + str(i)] / (1 - p.power(beta1, t))
                S_dw_corrected = self.s['S_dW' + str(i)] / (1 - p.power(beta2, t))
                S_db_corrected = self.s['S_db' + str(i)] / (1 - p.power(beta2, t))

                layer.W -= learning_rate * V_dw_corrected / (p.sqrt(S_dw_corrected) + epsilon)
                layer.b -= learning_rate * V_db_corrected / (p.sqrt(S_db_corrected) + epsilon)
                del layer.dW, layer.db

            if layer.batch_normal is not None:
                layer.batch_normal.dbeta, layer.batch_normal.dgamma = gradients_clip(layer.batch_normal.dbeta,
                                                                                     layer.batch_normal.dgamma)
                self.v['V_dbeta' + str(i)] = beta1 * self.v['V_dbeta' + str(i)] + (
                        1 - beta1) * layer.batch_normal.dbeta
                self.v['V_dgamma' + str(i)] = beta1 * self.v['V_dgamma' + str(i)] + (
                        1 - beta1) * layer.batch_normal.dgamma
                self.s['S_dbeta' + str(i)] = beta2 * self.s['S_dbeta' + str(i)] + (1 - beta2) * p.square(
                    layer.batch_normal.dbeta)
                self.s['S_dgamma' + str(i)] = beta2 * self.s['S_dgamma' + str(i)] + (1 - beta2) * p.square(
                    layer.batch_normal.dgamma)

                V_dbeta_corrected = self.v['V_dbeta' + str(i)] / (1 - p.power(beta1, t))
                V_dgamma_corrected = self.v['V_dgamma' + str(i)] / (1 - p.power(beta1, t))
                S_dbeta_corrected = self.s['S_dbeta' + str(i)] / (1 - p.power(beta2, t))
                S_dgamma_corrected = self.s['S_dgamma' + str(i)] / (1 - p.power(beta2, t))

                layer.batch_normal.beta -= learning_rate * V_dbeta_corrected / (p.sqrt(S_dbeta_corrected) + epsilon)
                layer.batch_normal.gamma -= learning_rate * V_dgamma_corrected / (
                        p.sqrt(S_dgamma_corrected) + epsilon)
                del layer.batch_normal.dbeta, layer.batch_normal.dgamma


class BatchGradientDescent(Optimize):
    def __init__(self, layers):
        super(BatchGradientDescent, self).__init__(layers)

    def update(self, t, learning_rate):
        for layer in self.layers:
            if isinstance(layer, Pooling):
                continue

            if isinstance(layer, SimpleRNN):
                layer.Waa -= learning_rate * layer.dWaa
                layer.Wax -= learning_rate * layer.dWax
                layer.Wya -= learning_rate * layer.dWya
                layer.b -= learning_rate * layer.db
                layer.by -= learning_rate * layer.dby
                del layer.dWaa, layer.dWax, layer.dWya, layer.db, layer.dby
            else:
                layer.dW, layer.db = gradients_clip(layer.dW, layer.db)
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db
                del layer.dW, layer.db

            if layer.batch_normal is not None:
                layer.batch_normal.dbeta, layer.batch_normal.dgamma = gradients_clip(layer.batch_normal.dbeta,
                                                                                     layer.batch_normal.dgamma)
                layer.batch_normal.beta -= learning_rate * layer.batch_normal.dbeta
                layer.batch_normal.gamma -= learning_rate * layer.batch_normal.dgamma
                del layer.batch_normal.dbeta, layer.batch_normal.dgamma


#  梯度裁剪
def gradients_clip(*args, theta=5.):
    grads = []
    if args:
        for gradient in args:
            grad = theta / (p.linalg.norm(gradient) + 1e-11) * gradient
            grads.append(grad)
    return grads
