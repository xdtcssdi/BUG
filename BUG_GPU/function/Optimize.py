import gc
import cupy as cp
from BUG_GPU.Layers.Layer import Convolution, Core


class Optimize:
    def __init__(self, layers):
        self.layers = layers

    def updata(self, *arg):
        pass


class Momentum(Optimize):
    def __init__(self, layers):
        super(Momentum, self).__init__(layers)
        self.v = {}
        self.s = {}
        for i in range(len(layers)):
            layer = self.layers[i]
            if isinstance(layer, Core) or isinstance(layer, Convolution):
                self.v['V_dW' + str(i)] = cp.zeros(layer.W.shape)
                self.v['V_db' + str(i)] = cp.zeros(layer.b.shape)
                if layer.batchNormal is not None:
                    self.v['V_dbeta' + str(i)] = cp.zeros(layer.batchNormal.dbeta.shape)
                    self.v['V_dgamma' + str(i)] = cp.zeros(layer.batchNormal.dgamma.shape)

    def updata(self, t, learning_rate, it, iterator, beta=0.9):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Core) or isinstance(layer, Convolution):
                self.v['V_dW' + str(i)] = beta * self.v['V_dW' + str(i)] + (1 - beta) * layer.dW
                self.v['V_db' + str(i)] = beta * self.v['V_db' + str(i)] + (1 - beta) * layer.db
                layer.W -= learning_rate * self.v['V_dW' + str(i)]
                layer.b -= learning_rate * self.v['V_db' + str(i)]
                del layer.dW, layer.db

                if layer.batchNormal is not None:
                    self.v['V_dbeta' + str(i)] = beta * self.v['V_dbeta' + str(i)] + (
                            1 - beta) * layer.batchNormal.dbeta
                    self.v['V_dgamma' + str(i)] = beta * self.v['V_dgamma' + str(i)] + (
                            1 - beta) * layer.batchNormal.dgamma
                    layer.batchNormal.beta -= learning_rate * self.v['V_dbeta' + str(i)]
                    layer.batchNormal.gamma -= learning_rate * self.v['V_dgamma' + str(i)]
                    del layer.batchNormal.dbeta, layer.batchNormal.dgamma
        gc.collect()


class Adam(Optimize):
    def __init__(self, layers):
        super(Adam, self).__init__(layers)
        self.v = {}
        self.s = {}
        for i in range(len(layers)):
            layer = self.layers[i]
            if isinstance(layer, (Core, Convolution)):
                self.v['V_dW' + str(i)] = cp.zeros(layer.W.shape)
                self.v['V_db' + str(i)] = cp.zeros(layer.b.shape)
                self.s['S_dW' + str(i)] = cp.zeros(layer.W.shape)
                self.s['S_db' + str(i)] = cp.zeros(layer.b.shape)
                if layer.batchNormal is not None:
                    self.v['V_dbeta' + str(i)] = cp.zeros(layer.batchNormal.dbeta.shape)
                    self.v['V_dgamma' + str(i)] = cp.zeros(layer.batchNormal.dgamma.shape)
                    self.s['S_dbeta' + str(i)] = cp.zeros(layer.batchNormal.dbeta.shape)
                    self.s['S_dgamma' + str(i)] = cp.zeros(layer.batchNormal.dgamma.shape)

    def updata(self, t, learning_rate, it, iterator, beta1=0.9, beta2=0.999, epsilon=1e-8):

        for i in range(len(self.layers)):
            layer = self.layers[i]
            if isinstance(layer, Core) or isinstance(layer, Convolution):

                self.v['V_dW' + str(i)] = beta1 * self.v['V_dW' + str(i)] + (1 - beta1) * layer.dW
                self.v['V_db' + str(i)] = beta1 * self.v['V_db' + str(i)] + (1 - beta1) * layer.db
                self.s['S_dW' + str(i)] = beta2 * self.s['S_dW' + str(i)] + (1 - beta2) * cp.square(layer.dW)
                self.s['S_db' + str(i)] = beta2 * self.s['S_db' + str(i)] + (1 - beta2) * cp.square(layer.db)
                V_dw_corrected = self.v['V_dW' + str(i)] / (1 - cp.power(beta1, t))
                V_db_corrected = self.v['V_db' + str(i)] / (1 - cp.power(beta1, t))
                S_dw_corrected = self.s['S_dW' + str(i)] / (1 - cp.power(beta2, t))
                S_db_corrected = self.s['S_db' + str(i)] / (1 - cp.power(beta2, t))

                layer.W -= learning_rate * V_dw_corrected / (cp.sqrt(S_dw_corrected) + epsilon)
                layer.b -= learning_rate * V_db_corrected / (cp.sqrt(S_db_corrected) + epsilon)

                del layer.dW, layer.db

                if layer.batchNormal is not None:
                    self.v['V_dbeta' + str(i)] = beta1 * self.v['V_dbeta' + str(i)] + (
                            1 - beta1) * layer.batchNormal.dbeta
                    self.v['V_dgamma' + str(i)] = beta1 * self.v['V_dgamma' + str(i)] + (
                            1 - beta1) * layer.batchNormal.dgamma
                    self.s['S_dbeta' + str(i)] = beta2 * self.s['S_dbeta' + str(i)] + (1 - beta2) * cp.square(
                        layer.batchNormal.dbeta)
                    self.s['S_dgamma' + str(i)] = beta2 * self.s['S_dgamma' + str(i)] + (1 - beta2) * cp.square(
                        layer.batchNormal.dgamma)

                    V_dbeta_corrected = self.v['V_dbeta' + str(i)] / (1 - cp.power(beta1, t))
                    V_dgamma_corrected = self.v['V_dgamma' + str(i)] / (1 - cp.power(beta1, t))
                    S_dbeta_corrected = self.s['S_dbeta' + str(i)] / (1 - cp.power(beta2, t))
                    S_dgamma_corrected = self.s['S_dgamma' + str(i)] / (1 - cp.power(beta2, t))

                    layer.batchNormal.beta -= learning_rate * V_dbeta_corrected / (cp.sqrt(S_dbeta_corrected) + epsilon)
                    layer.batchNormal.gamma -= learning_rate * V_dgamma_corrected / (
                            cp.sqrt(S_dgamma_corrected) + epsilon)
                    del layer.batchNormal.dbeta, layer.batchNormal.dgamma
        gc.collect()


class BatchGradientDescent(Optimize):
    def __init__(self, layers):
        super(BatchGradientDescent, self).__init__(layers)

    def updata(self, t, learning_rate, it, iterator):
        for layer in self.layers:
            if isinstance(layer, Core) or isinstance(layer, Convolution):
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db
                del layer.dW, layer.db
                if layer.batchNormal is not None:
                    layer.batchNormal.beta -= learning_rate * layer.batchNormal.dbeta
                    layer.batchNormal.gamma -= learning_rate * layer.batchNormal.dgamma
                    del layer.batchNormal.dbeta, layer.batchNormal.dgamma
        gc.collect()
