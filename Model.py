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
        self.cost = None

    def add(self, layer):
        assert (isinstance(layer, Layer))
        self.layers.append(layer)

    def getLayerNumber(self):
        return len(self.layers)

    def PartitionDataset(self, X, Y, testing_percentage, validation_percentage):
        total_m = X.shape[0]
        test_m = int(total_m * testing_percentage)
        vaild_m = int(total_m * validation_percentage)
        train_m = total_m - test_m - vaild_m

        X_train = X[:train_m]
        Y_train = Y[:train_m]

        X_test = X[train_m:train_m + test_m]
        Y_test = Y[train_m:train_m + test_m]

        X_valid = X[-vaild_m:]
        Y_valid = Y[-vaild_m:]

        return X_train, Y_train, X_test, Y_test, X_valid, Y_valid

    def train(self, X_train, Y_train, X_test, Y_test, batch_size, normalizing_inputs=True, testing_percentage=0.2,
              validation_percentage=0.2, learning_rate=0.075, iterator=2000,
              printLoss=False, lossMode='CrossEntry', tms=100, shuffle=True,
              printOneTime=False, log=sys.stdout,
              mode='train'):

        print("X_train.shape = %s, Y_train.shape = %s" % (X_train.shape, Y_train.shape))

        X_train = X_train.astype(np.float64)
        Y_train = Y_train.astype(np.float64)
        X_test = X_test.astype(np.float64)
        Y_test = Y_test.astype(np.float64)

        #  Normalizing inputs
        if normalizing_inputs:
            if len(X_train.shape) == 2:
                u = np.mean(X_train, axis=0)
                var = np.mean(X_train ** 2, axis=0)
                X_train -= u
                X_train /= var
                X_test -= u
                X_test /= var
            elif len(X_train.shape) > 2:
                X_train /= 255.0
                X_test /= 255.0
            else:
                raise ValueError
        #  Normalizing inputs

        #  shuffle start
        if shuffle:
            permutation = np.random.permutation(X_train.shape[0])
            X_train = X_train[permutation]
            Y_train = Y_train[permutation]
        #  shuffle end

        #  划分数据
        X_train, Y_train, _, __, X_valid, Y_valid = \
            self.PartitionDataset(X_train, Y_train, testing_percentage, validation_percentage)
        #  -------------

        # 初始化损失结构
        if lossMode == 'SoftmaxCrossEntry':
            self.cost = SoftCategoricalCross_entropy()
        else:
            self.cost = CrossEntry()

        start = 0
        if printOneTime:
            start = time.time()
        for it in range(iterator):
            in_cost = []

            # mini-batch
            for b in range(X_train.shape[0] // batch_size):
                bs = b * batch_size
                be = (b + 1) * batch_size
                x_train = X_train[bs:be]
                y_train = Y_train[bs:be]

                # 前向传播
                pre_A = x_train
                for layer in self.layers:
                    pre_A = layer.forward(pre_A, mode)

                # 损失计算
                loss = self.cost.forward(y_train, pre_A)
                in_cost.append(loss)
                # -------

                # 反向传播

                # 损失函数对最后一层Z的导数
                pre_grad = self.cost.backward(y_train, pre_A)
                for i in reversed(range(self.getLayerNumber())):
                    pre_grad = self.layers[i].backward(pre_grad)
                # -----------

                # 更新参数
                for layer in self.layers:
                    if isinstance(layer, Core) or isinstance(layer, Convolution):
                        layer.W -= learning_rate * layer.dW
                        layer.b -= learning_rate * layer.db
                        if layer.batchNormal is not None:
                            layer.batchNormal.beta -= learning_rate * layer.batchNormal.dbeta
                            layer.batchNormal.gamma -= learning_rate * layer.batchNormal.dgamma

            # ------------------
            if printLoss:
                if it % tms == 0:
                    cost = np.mean(in_cost)
                    print("iteration %d cost = %f" % (it, cost))
                    self.costs.append(cost)

        if printOneTime:
            print("consumer : ", time.time() - start)
        # 预测
        self.predict1(X_test, Y_test)

    def predict(self, X_train, Y_train):
        A = X_train
        for layer in self.layers:
            A = layer.forward(A)
        p = .0
        for i in range(X_train.shape[0]):
            t1 = self.returnMaxIdx(A[i])
            t2 = self.returnMaxIdx(Y_train[i])
            if t1 == t2:
                p += 1
        print("accuracy: %f%%" % (p / X_train.shape[0] * 100.))

    def predict1(self, X_train, Y_train):
        A = X_train
        for layer in self.layers:
            A = layer.forward(A, mode='test')
        p = .0
        for i in range(X_train.shape[0]):
            t1 = A[i][0] > 0.5
            t2 = Y_train[i][0]
            if t1 == t2:
                p += 1
        print("accuracy: %f%%" % (p / X_train.shape[0] * 100.))

    def returnMaxIdx(self, a):
        list_a = a.tolist()
        max_index = list_a.index(max(list_a))  # 返回最大值的索引
        return max_index

    def compile(self):
        for i in range(1, len(self.layers)):
            self.layers[i].pre_layer = self.layers[i - 1]
            self.layers[i - 1].next_layer = self.layers[i]
        self.layers[0].isFirst = True
        self.layers[-1].isLast = True
