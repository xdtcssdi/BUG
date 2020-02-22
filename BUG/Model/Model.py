import sys

from BUG.function import *
from BUG.Layers import Layer
import gc
from tqdm import trange

np.set_printoptions(threshold=np.inf)


class Model(object):

    def __init__(self):
        self.layers = []
        self.costs = []  # every batch cost
        self.cost = None  # 损失函数类
        self.optimize = None

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

    def normalizing_inputs(self, X_train, X_test, normalizing_inputs=True):
        if normalizing_inputs:
            if X_train.ndim == 2:
                u = np.mean(X_train, axis=0)
                var = np.mean(X_train ** 2, axis=0)
                X_train -= u
                X_train /= var
                X_test -= u
                X_test /= var
            elif X_train.ndim > 2:
                np.divide(X_train, 255.0, out=X_train, casting="unsafe")
                np.divide(X_test, 255.0, out=X_test, casting="unsafe")
            else:
                raise ValueError

    def train(self, X_train, Y_train, X_test, Y_test, batch_size, normalizing_inputs=True, testing_percentage=0.2,
              validation_percentage=0.2, learning_rate=0.075, iterator=2000,
              printLoss=False, lossMode='CrossEntry', shuffle=True,
              printOneTime=False, log=sys.stdout, optimize='BGD',
              mode='train'):
        assert not isinstance(X_train, np.float)
        assert not isinstance(X_test, np.float)
        t = 0

        print("X_train.shape = %s, Y_train.shape = %s" % (X_train.shape, Y_train.shape))

        #  Normalizing inputs
        self.normalizing_inputs(X_train, X_test, normalizing_inputs)
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

        costs = []

        #  mini_batch
        with trange(iterator) as tr:
            for it in tr:
                tr.set_description("第%d代:" % (it + 1))
                cost = self.mini_batch(X_train, Y_train, mode, learning_rate, batch_size, t, optimize)
                tr.set_postfix(batch_size=batch_size, loss=cost, acc=self.predict1(X_test, Y_test))
                costs.append(cost)

    def predict(self, X_train, Y_train):
        A = X_train
        for layer in self.layers:
            A = layer.forward(A, mode='test')
        p = .0
        for i in range(X_train.shape[0]):
            t1 = self.returnMaxIdx(A[i])
            t2 = self.returnMaxIdx(Y_train[i])
            if t1 == t2:
                p += 1
        return p * 1.0 / X_train.shape[0]

    def predict1(self, A, Y_train):
        for layer in self.layers:
            A = layer.forward(A, mode='test')
        p = .0
        for i in range(A.shape[0]):
            t1 = A[i][0] > 0.5
            t2 = Y_train[i][0]
            if t1 == t2:
                p += 1
        return p * 1.0 / A.shape[0]

    def returnMaxIdx(self, a):
        list_a = a.tolist()
        max_index = list_a.index(max(list_a))  # 返回最大值的索引
        return max_index

    def compile(self):
        for i in range(1, self.getLayerNumber()):
            self.layers[i].pre_layer = self.layers[i - 1]
            self.layers[i - 1].next_layer = self.layers[i]
        self.layers[0].isFirst = True
        self.layers[-1].isLast = True

    def train_step(self, x_train, y_train, mode, learning_rate, t, optimize):
        # 前向传播
        pre_A = x_train
        for layer in self.layers:
            pre_A = layer.forward(pre_A, mode)
        gc.collect()

        # 损失计算
        loss = self.cost.forward(y_train, pre_A)
        # -------

        # 反向传播

        #  损失函数对最后一层Z的导数
        pre_grad = self.cost.backward(y_train, pre_A)
        for layer in reversed(self.layers):
            pre_grad = layer.backward(pre_grad)
        gc.collect()
        # -----------

        #  更新参数
        if self.optimize is None:
            if optimize == 'Adam':
                self.optimize = Optimize.Adam(self.layers)
            elif optimize == 'Momentum':
                self.optimize = Optimize.Momentum(self.layers)
            elif optimize == 'BGD':
                self.optimize = Optimize.BatchGradientDescent(self.layers)
            else:
                raise ValueError

        t += 1
        self.optimize.updata(t, learning_rate)

        return loss

    def mini_batch(self, X_train, Y_train, mode, learning_rate, batch_size, t, optimize):
        # mini-batch
        in_cost = []
        num_complete = X_train.shape[0] // batch_size
        with trange(num_complete) as tr:
            for b in tr:
                bs = b * batch_size
                be = (b + 1) * batch_size
                x_train = X_train[bs:be]
                y_train = Y_train[bs:be]
                cost = self.train_step(x_train, y_train, mode, learning_rate, t, optimize)
                tr.set_postfix(loss=cost, acc=self.predict1(x_train, y_train))
                in_cost.append(cost)
            cost = self.train_step(X_train[num_complete * batch_size:], Y_train[num_complete * batch_size:],
                                   mode, learning_rate, t, optimize)
            tr.set_postfix(loss=cost, acc=self.predict1(x_train, y_train))
            in_cost.append(cost)

        return np.mean(in_cost)

    def summary(self):
        for i in range(0, len(self.layers)-1):
            layer = self.layers[i]
            if isinstance(layer, Core) or isinstance(layer, Convolution):
                print(layer.name + ' -> ' + layer.activation + ' -> ', end='')
            else:
                print(layer.name + ' -> ', end='')
        layer = self.layers[-1]
        if isinstance(layer, Core) or isinstance(layer, Convolution):
            print(layer.name + ' -> ' + layer.activation + ' -> ', end='')
        else:
            print(layer.name + ' -> ', end='')
        print('y_hat')
