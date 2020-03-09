import os.path
import pickle

import goto
import numpy as np
from goto import with_goto
from tqdm import trange, tqdm

from BUG.Layers.Layer import Layer, Dense, Convolution, LSTM, Embedding, Pooling, generate_layer
from BUG.function import Optimize
from BUG.function.Loss import SoftCategoricalCross_entropy
from BUG.function.util import minibatch, decode_captions
from BUG.load_package import p


class Linear_model(object):

    def __init__(self):
        self.layers = []
        self.costs = []  # every batch cost
        self.cost = None  # 损失函数类
        self.optimizer = None
        self.evaluate = None
        self.ndim = 2
        self.optimizeMode = None
        self.accuracy = None
        self.permutation = None

    def add(self, layer):
        assert isinstance(layer, Layer) or isinstance(layer, list) or isinstance(layer, tuple), '类型错误'
        self.layers.append(layer)

    def getLayerNumber(self):
        return len(self.layers)

    # 划分数据
    def partitionDataset(self, X, Y, testing_percentage, validation_percentage):
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
    def normalizing_inputs(self, X_train, X_test, ep=1e-11):
        if X_train.ndim == 2:
            self.ndim = 2
            self.u = p.mean(X_train, axis=0) + ep
            self.var = p.mean(X_train ** 2, axis=0) + ep

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
        return X_train, X_test

    # 训练
    @with_goto
    def fit(self, X_train, Y_train, X_test=None, Y_test=None, batch_size=15, testing_percentage=0.2,
            validation_percentage=0.2, learning_rate=0.075, iterator=2000, save_epoch=10,
            shuffle=True, mode='train', filename='train_params',
            path='data', regularization='L2', lambd=0):
        assert not isinstance(X_train, p.float)
        assert not isinstance(X_test, p.float)
        print("X_train.shape = %s, type = %s" % (X_train.shape, type(X_train)))
        print("Y_train.shape = %s, type = %s" % (Y_train.shape, type(Y_train)))
        self.args = {'batch_size': batch_size, 'testing_percentage': testing_percentage,
                     'validation_percentage': validation_percentage, 'learning_rate': learning_rate,
                     'iterator': iterator, 'save_epoch': save_epoch, 'shuffle': shuffle, 'mode': mode,
                     'filename': filename, 'path': path, 'regularization': regularization, 'lambd': lambd}
        t = 0
        start_it = 0
        if not os.path.exists(path):
            os.mkdir(path)

        if os.path.isfile(path + os.sep + 'caches.npz'):
            with open(path + os.sep + 'caches.obj', 'rb+') as f:
                start_it, t = pickle.load(f)
            self.permutation = p.load(path + os.sep + 'caches.npz')['permutation']
            self.load_model(path, filename)

        #  Normalizing inputs
        if self.is_normalizing:
            X_train, X_test = self.normalizing_inputs(X_train, X_test)
        #  Normalizing inputs

        #  shuffle start
        if shuffle:
            if not os.path.isfile(path + os.sep + 'caches.npz'):
                self.permutation = np.random.permutation(X_train.shape[0])

            X_train = X_train[self.permutation]
            Y_train = Y_train[self.permutation]
        #  shuffle end

        #  划分数据
        if X_test is None and Y_test is None:
            X_train, Y_train, X_test, Y_test, X_valid, Y_valid = \
                self.partitionDataset(X_train, Y_train, testing_percentage, validation_percentage)
        #  -------------

        is_continue = False  # flag

        label.point
        try:
            with trange(start_it, iterator) as tr:
                for self.it in tr:
                    tr.set_description("第%d代:" % (self.it + 1))
                    train_loss = self.mini_batch(X_train, Y_train, mode, learning_rate, batch_size, t,
                                                 regularization, lambd)
                    test_cost, acc = self.accuracy(X_test, Y_test, self.layers)
                    tr.set_postfix(batch_size=batch_size, train_loss=train_loss, test_loss=test_cost, acc=acc)
                    if self.it != 0 and self.it % save_epoch == 0:
                        self.interrupt(path, self.permutation, self.it, t)
                        self.save_model(path, filename)
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
            goto.point

    # 中断处理
    def interrupt(self, path, permutation, start_it, t):
        with open(path + os.sep + 'caches.obj', 'wb+') as f:
            pickle.dump((start_it, t), f)
        p.savez_compressed(path + os.sep + 'caches.npz', permutation=permutation)

    def compute_reg_loss(self, m, regularization='L2', lambd=0.1):
        """
        附加惩罚的loss
        :param m: batch_size
        :param regularization: 正则化模式
        :param lambd: 超参数lambd
        :return:
        """
        if lambd == 0:
            return 0

        reg_loss = .0
        if regularization == 'L2':
            for layer in self.layers:
                if not isinstance(Layer, Pooling):
                    reg_loss += p.sum(p.square(layer.parameters['W']))
        elif regularization == 'L1':
            for layer in self.layers:
                if not isinstance(Layer, Pooling):
                    reg_loss += p.sum(p.abs(layer.parameters['W']))
        else:
            raise ValueError

        return reg_loss * lambd / (2 * m)

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
            x = layer.forward(x, None, mode='test')
        return x

    # 组合层级关系
    def compile(self, lossMode, optimize, accuracy, is_normalizing=False):
        self.is_normalizing = is_normalizing
        self.accuracy = accuracy
        # 优化模式 str
        self.optimizeMode = optimize
        # 初始化损失结构
        self.cost = lossMode

        for i in range(1, self.getLayerNumber()):
            self.layers[i].pre_layer = self.layers[i - 1]
            self.layers[i - 1].next_layer = self.layers[i]
        self.layers[0].isFirst = True
        self.layers[-1].isLast = True

    # 单步训练
    def train_step(self, x_train, y_train, mode, learning_rate, t, regularization, lambd):
        # 前向传播
        batch_size = x_train.shape[0]
        output = x_train
        for layer in self.layers:
            output = layer.forward(output, None, mode)
        # 损失计算
        loss = self.cost.forward(y_train, output) + self.compute_reg_loss(batch_size, regularization, lambd)
        # -------

        # 反向传播

        #  损失函数对最后一层Z的导数
        dout = self.cost.backward(y_train, output)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        # -----------

        # 添加正则惩罚loss的梯度
        if lambd > .0:
            for layer in self.layers:
                if not isinstance(Layer, Pooling):
                    layer.gradients['W'] += lambd / batch_size * layer.parameters['W']

        if self.optimizer is None:
            if self.optimizeMode == 'Adam':
                self.optimizer = Optimize.Adam(self.layers)
            elif self.optimizeMode == 'Momentum':
                self.optimizer = Optimize.Momentum(self.layers)
            elif self.optimizeMode == 'BGD':
                self.optimizer = Optimize.BatchGradientDescent(self.layers)
            else:
                raise ValueError

        #  更新参数
        t += 1
        self.optimizer.update(t, learning_rate)

        return loss

    # mini-batch
    def mini_batch(self, X_train, Y_train, mode, learning_rate, batch_size, t, regularization, lambd):
        in_cost = []
        num_complete = X_train.shape[0] // batch_size
        with trange(num_complete) as tr:
            for b in tr:
                bs = b * batch_size
                be = (b + 1) * batch_size
                x_train = X_train[bs:be]
                y_train = Y_train[bs:be]
                cost = self.train_step(x_train, y_train, mode, learning_rate, t, regularization, lambd)
                tr.set_postfix(loss=cost)
                in_cost.append(cost)

            s = num_complete * batch_size
            if s < X_train.shape[0]:
                cost = self.train_step(X_train[num_complete * batch_size:], Y_train[num_complete * batch_size:],
                                       mode, learning_rate, t, regularization, lambd)
                tr.set_postfix(loss=cost)
                in_cost.append(cost)

        return sum(in_cost) / len(in_cost)

    # 网络详情
    def summary(self):
        for i in range(len(self.layers) - 1):
            layer = self.layers[i]
            if isinstance(layer, Dense) or isinstance(layer, Convolution):
                print(layer.name + ' -> ' + layer.activation + ' -> ', end='')
            else:
                print(layer.name + ' -> ', end='')
        layer = self.layers[-1]
        if isinstance(layer, Dense) or isinstance(layer, Convolution):
            print(layer.name + ' -> ' + layer.activation + ' -> ', end='')
        else:
            print(layer.name + ' -> ', end='')
        print('y_hat')

    # 保存模型参数
    def save_model(self, path, filename):
        layers = []
        for layer in self.layers:
            name = layer.save_params(path, filename)
            layers.append(name)

        with open(path + os.sep + filename + '.obj', 'wb') as f:
            pickle.dump(layers, f)
            pickle.dump(self.optimizeMode, f)
            pickle.dump(self.is_normalizing, f)
            pickle.dump(self.ndim, f)
            pickle.dump(self.accuracy, f)
            pickle.dump(self.cost, f)
        if self.is_normalizing and self.ndim == 2:
            p.savez_compressed(path + os.sep + filename + '_normalize.npz', u=self.u, var=self.var)

    #  加载模型参数
    def load_model(self, path, filename):
        Dense.count=0
        Convolution.count=0
        Pooling.count=0
        self.layers.clear()
        with open(path + os.sep + filename + '.obj', 'rb') as f:
            layers = pickle.load(f)
            for layer_name in layers:
                with open(path + os.sep + layer_name + '_' + filename + '_struct.obj', 'rb') as ff:
                    self.layers.append(generate_layer(layer_name.split('_')[0], pickle.load(ff)))

            for layer in self.layers:
                layer.load_params(path, filename)

            self.optimizeMode = pickle.load(f)
            if self.optimizeMode == 'Adam':
                self.optimizer = Optimize.Adam(self.layers)
            elif self.optimizeMode == 'Momentum':
                self.optimizer = Optimize.Momentum(self.layers)
            elif self.optimizeMode == 'BGD':
                self.optimizer = Optimize.BatchGradientDescent(self.layers)
            else:
                raise ValueError

            self.is_normalizing = pickle.load(f)
            self.ndim = pickle.load(f)
            self.accuracy = pickle.load(f)
            self.cost = pickle.load(f)
        if self.is_normalizing and self.ndim == 2:
            r = p.load(path + os.sep + filename + '_normalize.npz')
            self.u = r['u']
            self.var = r['var']


class LSTM_model(object):
    def __init__(self, hidden_dim, word_to_idx):

        self.costs = []  # every batch cost
        self.cost = None  # 损失函数类
        self.optimizer = None
        self.optimizeMode = None

        self.A0_layer = Dense(unit_number=hidden_dim, activation=None)  # a0输入
        self.X_layer = Embedding(vocab_size=len(word_to_idx), word_dim=256)  # X
        self.lstm_layer = LSTM(n_a=hidden_dim, word_to_idx=word_to_idx)  # 返回a
        self.output_layer = Dense(unit_number=len(word_to_idx), activation='softmax')
        self.layers = [self.A0_layer, self.X_layer, self.lstm_layer, self.output_layer]

    # 训练
    @with_goto
    def fit(self, data, batch_size=15, learning_rate=0.075, iterator=2000, optimize='Adam',
            save_epoch=10, filename='train_params', path='mnist_dnn_parameters'):

        if not os.path.exists(path):
            os.mkdir(path)

        self.cost = SoftCategoricalCross_entropy()
        start_it = 0
        if os.path.isfile(path + os.sep + 'caches.obj'):
            with open(path + os.sep + 'caches.obj', 'rb+') as f:
                start_it = pickle.load(f)
            self.load_model(path, filename)

        is_continue = False
        label.point
        try:
            with trange(start_it, iterator) as tr:
                for self.it in tr:
                    cost = []
                    with tqdm(minibatch(data, batch_size=batch_size)) as batch_data:

                        for captions_in, captions_out, features, urls in batch_data:

                            a0 = self.A0_layer.forward(features)
                            embedding_out = self.X_layer.forward(captions_in)

                            lstm_out = self.lstm_layer.forward(embedding_out, a0)

                            y_hat = self.output_layer.forward(lstm_out)

                            loss = self.cost.forward(captions_out, y_hat)
                            cost.append(loss)
                            batch_data.set_postfix(loss=loss)
                            dout = self.cost.backward(captions_out, y_hat)

                            dlstm = self.output_layer.backward(dout)

                            dembedding_out, da0 = self.lstm_layer.backward(dlstm)

                            self.X_layer.backward(dembedding_out)
                            self.A0_layer.backward(da0)

                            #  更新参数

                            if self.optimizer is None:
                                if optimize == 'Adam':
                                    self.optimizer = Optimize.Adam(self.layers)
                                elif optimize == 'Momentum':
                                    self.optimizer = Optimize.Momentum(self.layers)
                                elif optimize == 'BGD':
                                    self.optimizer = Optimize.BatchGradientDescent(self.layers)
                                else:
                                    raise ValueError

                            self.optimizer.update(self.it + 1, learning_rate)

                    if self.it and self.it % save_epoch == 0:
                        print(cost)
                        self.interrupt(path, self.it)
                        self.save_model(path, filename)

                    tr.set_postfix(loss=sum(cost) / len(cost))
            self.predict(data)

        except KeyboardInterrupt:
            c = input('请输入(Y)保存模型以便继续训练,(C) 继续执行 :')
            if c == 'Y' or c == 'y':
                self.interrupt(path, self.it)
                self.save_model(path, filename)
                print('已经中断训练。\n再次执行程序，继续从当前开始执行。')
            elif c == 'C' or c == 'c':
                is_continue = True
            else:
                print('结束执行')
        if is_continue:
            start_it = self.it
            is_continue = False
            goto.point

    # 中断处理
    def interrupt(self, path, start_it):
        with open(path + os.sep + 'caches.obj', 'wb+') as f:
            pickle.dump(start_it, f)

    def sample(self, features, layers, max_length=50):
        d1, e1, l1, d2 = layers
        N = features.shape[0]
        captions = l1.null_code * p.ones((N, max_length), dtype=p.int32)

        N, D = features.shape
        affine_out = d1.forward(features)

        prev_word_idx = [l1.start_code] * N
        prev_h = affine_out
        prev_c = p.zeros(prev_h.shape)
        captions[:, 0] = l1.start_code
        for i in range(1, max_length):
            prev_word_embed = e1.parameters['W'][prev_word_idx]
            next_h, next_c, cache = l1.lstm_step_forward(prev_word_embed, prev_h, prev_c)
            prev_c = next_c
            vocab_affine_out = d2.forward(next_h.reshape(-1, 1, 512))
            captions[:, i] = p.array(p.argmax(vocab_affine_out, axis=1))
            prev_word_idx = captions[:, i]
            prev_h = next_h

        return captions

    def predict(self, data):
        for split in ['train', 'val']:
            gt_captions, gt_captions_out, features, urls = list(minibatch(data, split=split, batch_size=2))[0]

            gt_captions = decode_captions(gt_captions, data['idx_to_word'])

            sample_captions = self.sample(features, self.layers)
            sample_captions = decode_captions(sample_captions, data['idx_to_word'])

            for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
                print(url)
                print('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))

    # 保存模型参数
    def save_model(self, path, filename):
        for layer in self.layers:
            layer.save_params(path, filename)

    # 加载模型参数
    def load_model(self, path, filename):

        for layer in self.layers:
            layer.load_params(path, filename)
