import gc
import os.path
import pickle

import numpy as np
from tqdm import trange

from BUG.Layers.Layer import Layer, Core, Convolution, Pooling
from BUG.function import Optimize
from BUG.function.Loss import SoftCategoricalCross_entropy, CrossEntry
from BUG.load_package import p
from goto import with_goto


class Model(object):

    def __init__(self):
        self.layers = []
        self.costs = []  # every batch cost
        self.cost = None  # 损失函数类
        self.optimizer = None
        self.evaluate = None
        self.ndim = 2
        self.optimizeMode = None

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
    def fit(self, X_train, Y_train, X_test=None, Y_test=None, batch_size=15, is_normalizing=True, testing_percentage=0.2,
            validation_percentage=0.2, learning_rate=0.075, iterator=2000, save_epoch=10,
            lossMode='CrossEntry', shuffle=True, optimize='BGD', mode='train', start_it=0, filename='model', path='data'):
        assert not isinstance(X_train, p.float)
        assert not isinstance(X_test, p.float)
        print("X_train.shape = %s, Y_train.shape = %s" % (X_train.shape, Y_train.shape))
        print("X_train.type = %s, Y_train.type = %s" % (type(X_train), type(Y_train)))
        t = 0
        self.optimizeMode = optimize
        if not os.path.exists(path):
            os.mkdir(path)

        if os.path.isfile(path+ os.sep + 'caches.npz'):
            with open(path+ os.sep + 'caches.npz', 'rb+') as f:
                r = p.load(path+ os.sep +'caches.npz')
                start_it = r['start_it']
                t = r['t']
                self.permutation = r['permutation']

            self.load_model(path, filename)

        #  Normalizing inputs
        self.is_normalizing = is_normalizing
        self.normalizing_inputs(X_train, X_test, is_normalizing)
        #  Normalizing inputs

        #  shuffle start
        if shuffle:
            if not os.path.isfile(path+ os.sep + 'caches.npz'):
                self.permutation = np.random.permutation(X_train.shape[0])

            X_train = X_train[self.permutation]
            Y_train = Y_train[self.permutation]
        #  shuffle end

        #  划分数据
        if X_test is None and Y_test is None:
            X_train, Y_train, X_test, Y_test, X_valid, Y_valid = \
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
                    cost = self.mini_batch(X_train, Y_train, mode, learning_rate, batch_size, t, self.it,
                                           iterator, optimize)
                    tr.set_postfix(batch_size=batch_size, loss=cost, acc=self.evaluate(X_test, Y_test))
                    if self.it != 0 and self.it % save_epoch == 0:
                        self.interrupt(path, self.permutation, self.it, t)
                        self.save_model(path, filename)
                    costs.append(cost)
        except KeyboardInterrupt:
            c = input('请输入(Y)保存模型以便继续训练,(C) 继续执行 :')
            if c == 'Y' or c == 'y':
                self.interrupt(path, self.permutation, self.it, t)
                self.save_model(path, filename)
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
    def interrupt(self, path, permutation, start_it, t):
        with open(path + os.sep + 'caches.npz', 'wb') as f:
            p.savez_compressed(path + os.sep + 'caches.npz', permutation=permutation, start_it=start_it, t=t)

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
    def train_step(self, x_train, y_train, mode, learning_rate, t, it, iterator, optimize):
        # 前向传播
        output = x_train
        for layer in self.layers:
            output = layer.forward(output, mode)

        # 损失计算
        loss = self.cost.forward(y_train, output)
        # -------

        # 反向传播

        #  损失函数对最后一层Z的导数
        dout = self.cost.backward(y_train, output)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        # -----------

        if self.optimizer is None:
            if optimize == 'Adam':
                self.optimizer = Optimize.Adam(self.layers)
            elif optimize == 'Momentum':
                self.optimizer = Optimize.Momentum(self.layers)
            elif optimize == 'BGD':
                self.optimizer = Optimize.BatchGradientDescent(self.layers)
            else:
                raise ValueError

        #  更新参数
        gc.collect()
        t += 1
        self.optimizer.update(t, learning_rate, it, iterator)

        return loss

    # mini-batch
    def mini_batch(self, X_train, Y_train, mode, learning_rate, batch_size, t, it, iterator, optimize):
        in_cost = []
        num_complete = X_train.shape[0] // batch_size
        with trange(num_complete) as tr:
            for b in tr:
                bs = b * batch_size
                be = (b + 1) * batch_size
                x_train = X_train[bs:be]
                y_train = Y_train[bs:be]
                cost = self.train_step(x_train, y_train, mode, learning_rate, t, it, iterator, optimize)
                tr.set_postfix(loss=cost)
                in_cost.append(cost)

            s = num_complete * batch_size
            if s < X_train.shape[0]:
                cost = self.train_step(X_train[num_complete * batch_size:], Y_train[num_complete * batch_size:],
                                       mode, learning_rate, t, it, iterator, optimize)
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
    def save_model(self,path, filename):
        for layer in self.layers:
            layer.save_params(path, filename)

        with open(path + os.sep + filename+'.obj', 'wb') as f:
            pickle.dump(self.optimizeMode, f)
            pickle.dump(self.evaluate, f)
            pickle.dump(self.is_normalizing, f)
            pickle.dump(self.ndim, f)
            if self.is_normalizing and self.ndim == 2:
                p.savez_compressed(path + os.sep + filename + '_normalize.npz', u=self.u, var=self.var)

    # 加载模型参数
    def load_model(self,path, filename):

        for layer in self.layers:
            layer.load_params(path, filename)

        with open(path+ os.sep + filename+'.obj', 'rb') as f:
            self.optimizeMode = pickle.load(f)
            if self.optimizeMode == 'Adam':
                self.optimizer = Optimize.Adam(self.layers)
            elif self.optimizeMode == 'Momentum':
                self.optimizer = Optimize.Momentum(self.layers)
            elif self.optimizeMode == 'BGD':
                self.optimizer = Optimize.BatchGradientDescent(self.layers)
            else:
                raise ValueError

            self.evaluate = pickle.load(f)
            self.is_normalizing = pickle.load(f)
            self.ndim = pickle.load(f)
            if self.is_normalizing and self.ndim == 2:
                r = p.savez_compressed(path + os.sep + filename + '_normalize.npz', u=self.u, var=self.var)
                self.u = r['u']
                self.var = r['var']
