import gc
import os.path
import pickle

import numpy as np
from tqdm import trange

from BUG.Layers.Layer import Layer, Core, Convolution
from BUG.function import Optimize
from BUG.function.Loss import SoftCategoricalCross_entropy, CrossEntry
from BUG.load_package import p
from goto import with_goto


class Model(object):

    def __init__(self):
        self.layers = []
        self.costs = []  # every batch cost
        self.cost = None  # 损失函数类
        self.optimize = None
        self.evaluate = None
        self.ndim = 2

    def add(self, layer):
        assert (isinstance(layer, Layer))
        self.layers.append(layer)

    def getLayerNumber(self):
        return len(self.layers)

    # 划分数据
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

    # 归一化输入
    def normalizing_inputs(self, X_train, X_test, is_normalizing=True):
        if is_normalizing:
            if X_train.ndim == 2:
                self.ndim = 2
                self.u = p.mean(X_train, axis=0)
                self.var = p.mean(X_train ** 2, axis=0)
                X_train -= self.u
                X_train /= self.var
                X_test -= self.u
                X_test /= self.var
            elif X_train.ndim > 2:
                self.ndim = X_train.ndim
                p.divide(X_train, 255.0, out=X_train, casting="unsafe")
                p.divide(X_test, 255.0, out=X_test, casting="unsafe")
            else:
                raise ValueError

    # 训练
    @with_goto
    def fit(self, X_train, Y_train, X_test, Y_test, batch_size, is_normalizing=True, testing_percentage=0.2,
            validation_percentage=0.2, learning_rate=0.075, iterator=2000,
            lossMode='CrossEntry', shuffle=True, optimize='BGD', mode='train', start_it=0, filename='model.h5'):
        assert not isinstance(X_train, p.float)
        assert not isinstance(X_test, p.float)
        print("X_train.shape = %s, Y_train.shape = %s" % (X_train.shape, Y_train.shape))
        print("X_train.type = %s, Y_train.type = %s" % (type(X_train), type(Y_train)))
        t = 0

        if os.path.isfile('caches.data'):
            with open('caches.data', 'rb+') as f:
                data = pickle.load(f)
                self.permutation, start_it, t = data
            self.load_model(filename)

        #  Normalizing inputs
        self.is_normalizing = is_normalizing
        self.normalizing_inputs(X_train, X_test, is_normalizing)
        #  Normalizing inputs

        #  shuffle start
        if shuffle:
            if not os.path.isfile('caches.data'):
                self.permutation = np.random.permutation(X_train.shape[0])

            X_train = X_train[self.permutation]
            Y_train = Y_train[self.permutation]
        #  shuffle end

        #  划分数据
        X_train, Y_train, _, __, X_valid, Y_valid = \
            self.PartitionDataset(X_train, Y_train, testing_percentage, validation_percentage)
        #  -------------

        # 初始化损失结构
        if lossMode == 'SoftmaxCrossEntry':
            self.cost = SoftCategoricalCross_entropy()
            self.evaluate = self.evaluate_many
        elif lossMode == 'CrossEntry':
            self.cost = CrossEntry()
            self.evaluate = self.evaluate_one
        else:
            raise ValueError

        costs = []

        #  mini_batch
        is_continue = False
        label .point
        try:
            with trange(start_it, iterator) as tr:
                for self.it in tr:
                    tr.set_description("第%d代:" % (self.it + 1))
                    cost = self.mini_batch(X_train, Y_train, mode, learning_rate, batch_size, t, optimize, self.it,
                                           iterator)
                    tr.set_postfix(batch_size=batch_size, loss=cost, acc=self.evaluate(X_test, Y_test))
                    costs.append(cost)
        except KeyboardInterrupt:
            c = input('请输入(Y)保存模型以便继续训练,(C) 继续执行')
            if c == 'Y' or c == 'y':
                self.interrupt(self.permutation, self.it, t)
                self.save_model(filename)
                print('已经中断训练。\n再次执行程序，继续从当前开始执行。')
            elif c == 'C' or c == 'c':
                is_continue = True
            else:
                print('结束执行')
        if is_continue:
            start_it = self.it
            is_continue = False
            goto .point

    # 中断处理
    def interrupt(self, permutation, start_it, t):
        with open('caches.data', 'wb') as f:
            data = (permutation, start_it, t)
            pickle.dump(data, f)

    # 多输出评估
    def evaluate_many(self, X_train, Y_train):
        A = X_train
        for layer in self.layers:
            A = layer.forward(A, mode='test')
        return (p.argmax(A, -1) == p.argmax(Y_train, -1)).sum() / X_train.shape[0]

    # 单输出评估
    def evaluate_one(self, A, Y_train):
        for layer in self.layers:
            A = layer.forward(A, mode='test')
        return ((A > 0.5) == Y_train).sum() / A.shape[0]

    # 预测
    def predict(self, x):

        if self.is_normalizing:
            if x.ndim == 2:
                x -= self.u
                x /= self.var
            elif x.ndim > 2:
                p.divide(x, 255.0, out=x, casting="unsafe")
            else:
                raise ValueError

        for layer in self.layers:
            x = layer.forward(x, mode='test')
        return x

    # 组合层级关系
    def compile(self):
        for i in range(1, self.getLayerNumber()):
            self.layers[i].pre_layer = self.layers[i - 1]
            self.layers[i - 1].next_layer = self.layers[i]
        self.layers[0].isFirst = True
        self.layers[-1].isLast = True

    # 单步训练
    def train_step(self, x_train, y_train, mode, learning_rate, t, optimize, it, iterator):
        # 前向传播
        pre_A = x_train
        for layer in self.layers:
            pre_A = layer.forward(pre_A, mode)

        # 损失计算
        loss = self.cost.forward(y_train, pre_A)
        # -------

        # 反向传播

        #  损失函数对最后一层Z的导数
        pre_grad = self.cost.backward(y_train, pre_A)
        for layer in reversed(self.layers):
            pre_grad = layer.backward(pre_grad)
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
        gc.collect()
        t += 1
        self.optimize.updata(t, learning_rate, it, iterator)

        return loss

    # mini-batch
    def mini_batch(self, X_train, Y_train, mode, learning_rate, batch_size, t, optimize, it, iterator):
        in_cost = []
        num_complete = X_train.shape[0] // batch_size
        with trange(num_complete) as tr:
            for b in tr:
                bs = b * batch_size
                be = (b + 1) * batch_size
                x_train = X_train[bs:be]
                y_train = Y_train[bs:be]
                cost = self.train_step(x_train, y_train, mode, learning_rate, t, optimize, it, iterator)
                tr.set_postfix(loss=cost)
                in_cost.append(cost)

            s = num_complete * batch_size
            if s < X_train.shape[0]:
                cost = self.train_step(X_train[num_complete * batch_size:], Y_train[num_complete * batch_size:],
                                       mode, learning_rate, t, optimize, it, iterator)
                tr.set_postfix(loss=cost)
                in_cost.append(cost)

        return np.mean(in_cost)

    # 网络详情
    def summary(self):
        for i in range(len(self.layers) - 1):
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

    # 保存模型参数
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.optimize, f)
            pickle.dump(self.layers, f)
            pickle.dump(self.evaluate, f)
            pickle.dump(self.is_normalizing, f)
            pickle.dump(self.ndim, f)
            if self.is_normalizing and self.ndim == 2:
                pickle.dump(self.u, f)
                pickle.dump(self.var, f)

    # 加载模型参数
    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.optimize = pickle.load(f)
            self.layers = pickle.load(f)
            self.optimize.layers = self.layers
            self.evaluate = pickle.load(f)
            self.is_normalizing = pickle.load(f)
            self.ndim = pickle.load(f)
            if self.is_normalizing and self.ndim == 2:
                self.u = pickle.load(f)
                self.var = pickle.load(f)
