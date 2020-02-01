import numpy as np

from Loss import *
from Layers.Core import Layer


class Model(object):

    def __init__(self):
        self.layers = []

    def add(self, layer):
        assert (isinstance(layer, Layer))
        self.layers.append(layer)

    def getLayerNumber(self):
        return len(self.layers)

    def train(self, X_train, Y_train, learning_rate=0.075, iterator=2000, printLoss=False):

        for it in range(iterator):

            # 前向传播
            pre_A = X_train
            for layer in self.layers:
                pre_A = layer.forward(pre_A)

            # 打印损失
            loss = CrossEntry(Y_train, pre_A)
            if printLoss:
                if it % 100 == 0:
                    print("loss = ", loss)

            # 反向传播

            # 损失函数对最后一层Z的导数
            pre_grad = CrossEntryGrad(Y_train, pre_A)

            # 反向传播
            Lc = self.getLayerNumber()
            for i in reversed(range(Lc)):  # 1,0
                pre_grad, dW, db = self.layers[i].backward(pre_grad,
                        None if i == Lc-1 else self.layers[i+1].W,
                        last=True if i == Lc-1 else False)

            # 更新参数
            for layer in self.layers:
                W, b = layer.params
                dW, db = layer.grads

                layer.W = W - learning_rate * dW
                layer.b = b - learning_rate * db

    def complie(self,X_train):
        dims = [X_train.shape[0]]
        for layer in self.layers:
            dims.append(layer.unit_number)
        for i in range(len(dims)-1):
            self.layers[i].init_params(dims[i])