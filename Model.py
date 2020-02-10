import sys
import time
import numpy as np
sys.path.append('./Layers/')
from Loss import *
from Layers.Layer import *
from Layers.Core import Core
from Layers.Convolution import Convolution
np.set_printoptions(threshold=np.inf)


class Model(object):

    def __init__(self):
        self.layers = []
        self.costs = []

    def add(self, layer):
        assert (isinstance(layer, Layer))
        self.layers.append(layer)

    def getLayerNumber(self):
        return len(self.layers)

    def train(self, X_train, Y_train, batch_size=64, learning_rate=0.075, iterator=2000,
              printLoss=False, lossMode='CrossEntry', tms=100, shuffle=True, printOneTime=True, log=sys.stdout):
        print("X_train.shape = %s, Y_train.shape = %s" % (X_train.shape, Y_train.shape))
        start = 0
        for it in range(iterator):
            if printOneTime:
                start = time.time()
            # 前向传播
            pre_A = X_train
            for layer in self.layers:
                pre_A = layer.forward(pre_A)
            self.y_hat = pre_A
            # 打印损失
            if lossMode == 'SoftmaxCrossEntry':
                cost = SoftCategoricalCross_entropy()
            else:
                cost = CrossEntry()
            loss = cost.forward(Y_train, pre_A)
            if printLoss:
                if it % tms == 0:
                    print("iteration %d : cost = %s" % (it, loss))
                    self.costs.append(loss)

            # 反向传播

            # 损失函数对最后一层Z的导数
            pre_grad = cost.backward(Y_train, pre_A)

            # 反向传播
            for i in reversed(range(self.getLayerNumber())):
                pre_grad = self.layers[i].backward(pre_grad)
            # 更新参数
            for layer in self.layers:
                if isinstance(layer, Core) or isinstance(layer, Convolution):
                    layer.W -= learning_rate * layer.dW
                    layer.b -= learning_rate * layer.db
            if printOneTime:
                print("consumer : ", time.time()-start)

    def compile(self):
        for i in range(1, len(self.layers)):
            self.layers[i].pre_layer = self.layers[i-1]
            self.layers[i-1].next_layer = self.layers[i]
        self.layers[0].isFirst = True
        self.layers[-1].isLast = True


