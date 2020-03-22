import math
import os.path
import pickle

import goto
import numpy as np
from goto import with_goto
from tqdm import trange, tqdm

from BUG.Layers.Layer import Layer, Dense, Convolution, LSTM, Embedding, Pooling, generate_layer, SimpleRNN
from BUG.function import Optimize
from BUG.function.Loss import SoftCategoricalCross_entropy
from BUG.function.util import minibatch, decode_captions, one_hot, data_iter_consecutive
from BUG.load_package import p


class Sequentual(object):

    def __init__(self):
        self.layers = []
        self.accs=[]
        self.costs = []  # every batch cost
        self.cost = None  # 损失函数类
        self.optimizer = None
        self.optimizeMode = None
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
            mode='train', path='data', regularization='L2', lambd=0, is_print=False):
        self.args = {'batch_size': batch_size, 'testing_percentage': testing_percentage,
                     'validation_percentage': validation_percentage, 'learning_rate': learning_rate,
                     'iterator': iterator, 'save_epoch': save_epoch, 'mode': mode,
                     'path': path, 'regularization': regularization, 'lambd': lambd}
        print_disable = not is_print

        start_it = 0
        if not os.path.exists(path):
            os.mkdir(path)

        if os.path.isfile(path + os.sep + 'caches.obj'):
            start_it = self.load_model(path)

        #  Normalizing inputs
        if self.is_normalizing:
            X_train, X_test = self.normalizing_inputs(X_train, X_test)
        #  Normalizing inputs

        #  划分数据
        if X_test is None and Y_test is None:
            X_train, Y_train, X_test, Y_test, X_valid, Y_valid = \
                self.partitionDataset(X_train, Y_train, testing_percentage, validation_percentage)
        #  -------------

        is_continue = False  # flag

        label.point
        try:
            with trange(iterator, initial=start_it) as tr:
                for self.it in tr:
                    tr.set_description("第%d代:" % (self.it + 1))
                    train_loss, train_acc = self.mini_batch(X_train, Y_train, mode, learning_rate, batch_size, self.it,
                                                 regularization, lambd, print_disable)
                    val_cost, val_acc = self.accuracy(X_test, Y_test, self.layers)
                    self.costs.append([train_loss, val_cost])
                    self.accs.append([train_acc , val_acc])
                    tr.set_postfix(batch_size=batch_size, train_loss=train_loss, train_acc=train_acc,
                                   val_loss=val_cost, val_acc=val_acc)
                    if (self.it + 1) % save_epoch == 0:
                        self.save_model(path, self.it)
        except KeyboardInterrupt:
            c = input('请输入(Y)保存模型以便继续训练,(C) 继续执行 :')
            if c == 'Y' or c == 'y':
                self.save_model(path, self.it)
                print('已经中断训练。\n再次执行程序，继续从当前开始执行。')
            elif c == 'C' or c == 'c':
                is_continue = True
            else:
                print('结束执行')
        if is_continue:
            start_it = self.it
            is_continue = False
            goto.point

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
                if not isinstance(layer, Pooling):
                    reg_loss += p.sum(p.square(layer.parameters['W']))
        elif regularization == 'L1':
            for layer in self.layers:
                if not isinstance(layer, Pooling):
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
        # 精度计算
        train_acc = self.accuracy(output, y_train, self.layers, True)
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
                if not isinstance(layer, Pooling):
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
        self.optimizer.init_params(self.layers)
        self.optimizer.update(t+1, learning_rate)
        return loss, train_acc

    # mini-batch
    def mini_batch(self, X_train, Y_train, mode, learning_rate, batch_size, t, regularization, lambd,
                   print_disable=False):
        in_cost = []
        in_acc = []
        num_complete = X_train.shape[0] // batch_size
        with trange(num_complete, disable=print_disable) as tr:
            for b in tr:
                bs = b * batch_size
                be = (b + 1) * batch_size
                permutation = np.random.permutation(batch_size)
                x_train = X_train[bs:be][permutation]
                y_train = Y_train[bs:be][permutation]

                cost, train_acc = self.train_step(x_train, y_train, mode, learning_rate, t, regularization, lambd)

                tr.set_postfix(loss=cost, acc=train_acc)
                in_cost.append(cost)
                in_acc.append(train_acc)

            s = num_complete * batch_size
            if s < X_train.shape[0]:
                permutation = np.random.permutation(X_train.shape[0] - num_complete * batch_size)

                cost, train_acc = self.train_step(X_train[num_complete * batch_size:][permutation],
                                       Y_train[num_complete * batch_size:][permutation],
                                       mode, learning_rate, t, regularization, lambd)
                tr.set_postfix(loss=cost, acc=train_acc)
                in_cost.append(cost)
                in_acc.append(train_acc)

        return sum(in_cost) / len(in_cost), sum(in_acc) / len(in_acc)

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
    def save_model(self, path, start_it):
        layers = []
        for layer in self.layers:
            name = layer.save_params(path)
            layers.append(name)

        with open(path + os.sep + 'caches.obj', 'wb') as f:
            pickle.dump(start_it, f)
            pickle.dump(layers, f)
            pickle.dump(self.optimizeMode, f)
            pickle.dump(self.is_normalizing, f)
            pickle.dump(self.ndim, f)
            pickle.dump(self.accuracy, f)
            pickle.dump(self.cost, f)
        if self.is_normalizing and self.ndim == 2:
            p.savez_compressed(path + os.sep + 'train_normalize.npz', u=self.u, var=self.var)
        self.optimizer.save_parameters(path)

    #  加载模型参数
    def load_model(self, path):
        Dense.count = 0
        Convolution.count = 0
        Pooling.count = 0
        self.layers.clear()

        with open(path + os.sep + 'caches.obj', 'rb') as f:
            start_it = pickle.load(f)
            layers = pickle.load(f)
            for layer_name in layers:
                with open(path + os.sep + layer_name + '_struct.obj', 'rb') as ff:
                    self.add(generate_layer(layer_name.split('_')[0], pickle.load(ff)))

            for i in range(1, self.getLayerNumber()):
                self.layers[i].pre_layer = self.layers[i - 1]
                self.layers[i - 1].next_layer = self.layers[i]
                self.layers[0].isFirst = True
                self.layers[-1].isLast = True

            for layer in self.layers:
                layer.load_params(path)

            self.optimizeMode = pickle.load(f)
            if self.optimizeMode == 'Adam':
                self.optimizer = Optimize.Adam(self.layers)
            elif self.optimizeMode == 'Momentum':
                self.optimizer = Optimize.Momentum(self.layers)
            elif self.optimizeMode == 'BGD':
                self.optimizer = Optimize.BatchGradientDescent(self.layers)
            else:
                raise ValueError
            self.optimizer.load_parameters(path)
            self.is_normalizing = pickle.load(f)
            self.ndim = pickle.load(f)
            self.accuracy = pickle.load(f)
            self.cost = pickle.load(f)
        if self.is_normalizing and self.ndim == 2:
            r = p.load(path + os.sep + 'train_normalize.npz')
            self.u = r['u']
            self.var = r['var']
        return start_it


class LSTM_model(object):
    def __init__(self, hidden_dim, word_to_idx, point):

        self.costs = []  # every batch cost
        self.cost = None  # 损失函数类
        self.optimizer = None
        self.optimizeMode = None

        self.A0_layer = Dense(unit_number=hidden_dim, activation='relu')  # a0输入
        self.X_layer = Embedding(vocab_size=len(word_to_idx), word_dim=256)  # 词向量
        self.lstm_layer = LSTM(n_a=hidden_dim, word_to_idx=word_to_idx, point=point)  # 返回a
        self.output_layer = Dense(unit_number=len(word_to_idx), activation='softmax')  # 预测
        self.layers = [self.A0_layer, self.X_layer, self.lstm_layer, self.output_layer]

    # 训练
    @with_goto
    def fit(self, data, batch_size=15, learning_rate=0.075, iterator=2000, optimize='Adam',
            save_epoch=10, path='data', is_print=False):
        print_disable = not is_print
        if not os.path.exists(path):
            os.mkdir(path)

        self.cost = SoftCategoricalCross_entropy()
        start_it = 0
        if os.path.isfile(path + os.sep + 'caches.obj'):
            with open(path + os.sep + 'caches.obj', 'rb+') as f:
                start_it = pickle.load(f)
            self.load_model(path)
        theta = 1e-2
        is_continue = False
        cur_it = 0
        in_bar = tqdm()
        bar = tqdm()
        label.point
        try:
            bar = trange(iterator, initial=start_it)

            for it in bar:
                cost = []
                for captions_in, captions_out, features, urls in tqdm(minibatch(data, batch_size=batch_size),
                                                                      disable=print_disable):

                    permutation = np.random.permutation(captions_in.shape[0])
                    captions_in = captions_in[permutation]
                    captions_out = captions_out[permutation]
                    features = features[permutation]

                    a0 = self.A0_layer.forward(features)

                    embedding_out = self.X_layer.forward(captions_in)

                    lstm_out = self.lstm_layer.forward(embedding_out, a0)

                    y_hat = self.output_layer.forward(lstm_out)

                    loss = self.cost.forward(captions_out, y_hat)
                    cost.append(loss)

                    dout = self.cost.backward(captions_out, y_hat)

                    dlstm = self.output_layer.backward(dout)

                    dembedding_out, da0 = self.lstm_layer.backward(dlstm)

                    self.X_layer.backward(dembedding_out)
                    self.A0_layer.backward(da0)

                    #  更新参数

                    if self.optimizer is None:
                        if optimize == 'Adam':
                            self.optimizer = Optimize.Adam(self.layers, theta)
                        elif optimize == 'Momentum':
                            self.optimizer = Optimize.Momentum(self.layers, theta)
                        elif optimize == 'BGD':
                            self.optimizer = Optimize.BatchGradientDescent(self.layers, theta)
                        else:
                            raise ValueError
                    self.optimizer.init_params(self.layers)
                    self.optimizer.update(it + 1, learning_rate)
                if len(cost) == 0:
                    continue

                if (it + 1) % save_epoch == 0:
                    self.interrupt(path, it)
                    self.save_model(path)
                    self.predict(data)

                bar.set_postfix(loss=sum(cost) / len(cost))
                cur_it = it

        except KeyboardInterrupt:
            c = input('请输入(Y)保存模型以便继续训练,(C) 继续执行 :')
            if c == 'Y' or c == 'y':
                self.interrupt(path, cur_it)
                self.save_model(path)
                print('已经中断训练。\n再次执行程序，继续从当前开始执行。')
            elif c == 'C' or c == 'c':
                is_continue = True
            else:
                print('结束执行')
        if is_continue:
            start_it = cur_it
            is_continue = False
            goto.point
        bar.close()
        in_bar.close()

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
            gt_captions, gt_captions_out, features, urls = minibatch(data, split=split, batch_size=2)[0]

            gt_captions = decode_captions(gt_captions, data['idx_to_word'])

            sample_captions = self.sample(features, self.layers)
            sample_captions = decode_captions(sample_captions, data['idx_to_word'])

            for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
                print(url)
                print('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))

    # 保存模型参数
    def save_model(self, path):
        for layer in self.layers:
            layer.save_params(path)

    # 加载模型参数
    def load_model(self, path):
        for layer in self.layers:
            layer.load_params(path)


class Char_RNN(object):
    def __init__(self, hidden_unit, vocab_size, char_to_ix, ix_to_char, word_dim, cell='rnn'):
        self.it = 0
        self.hidden_unit = hidden_unit
        self.vocab_size = vocab_size
        self.char_to_ix = char_to_ix
        self.ix_to_char = ix_to_char
        self.optimizeMode = None
        self.out_layer = Dense(vocab_size, activation='softmax')
        self.X_layer = Embedding(vocab_size, word_dim)
        if cell == 'rnn':
            self.rnn_layer = SimpleRNN(hidden_unit)
        elif cell == 'lstm':
            self.rnn_layer = LSTM(char_to_ix, [1, 1, 1], hidden_unit)
        self.optimizer = None
        self.layers = [self.rnn_layer, self.out_layer, self.X_layer]

    def compile(self, optimize='Adam', learning_rate=0.001):
        self.optimizeMode = optimize
        self.learning_rate = learning_rate

    @with_goto
    def fit(self, data, batch_size=32, path='data', num_steps=32, iterator=2000, pred_len=50, prefix=None,
            save_epoch=10, is_print=False):

        start_it = 0
        if not os.path.exists(path):
            os.mkdir(path)
        if os.path.isfile(path + os.sep + 'caches.obj'):
            start_it = self.load_model(path)

        print_disable = not is_print
        if prefix is None:
            prefix = []
        loss_obj = SoftCategoricalCross_entropy()
        theta = 1e-2
        is_continue = False
        label.point
        try:
            with trange(iterator, initial=start_it) as out_bar:
                for self.it in out_bar:
                    cost = []
                    in_bar = tqdm(data_iter_consecutive(data, batch_size, num_steps), disable=print_disable)
                    for X, Y in in_bar:
                        # 前向传播
                        state = p.zeros([batch_size, self.hidden_unit])  # a0
                        inputs = self.X_layer.forward(X)
                        # inputs = one_hot(X, self.vocab_size)

                        state = self.rnn_layer.forward(inputs, state)
                        outputs = self.out_layer.forward(state)  # softmax层

                        # 计算损失
                        target = Y.reshape(None, )
                        curr_loss = loss_obj.forward(target, outputs)
                        in_bar.set_postfix(perplexity=math.exp(curr_loss), loss=curr_loss)
                        cost.append(curr_loss)

                        # 反向传播
                        dout = loss_obj.backward(target, outputs)
                        dsoft_layer = self.out_layer.backward(dout)

                        dx, _ = self.rnn_layer.backward(dsoft_layer)
                        self.X_layer.backward(dx)
                        if self.optimizer is None:
                            if self.optimizeMode == 'Adam':
                                self.optimizer = Optimize.Adam(self.layers, theta)
                            elif self.optimizeMode == 'Momentum':
                                self.optimizer = Optimize.Momentum(self.layers, theta)
                            elif self.optimizeMode == 'BGD':
                                self.optimizer = Optimize.BatchGradientDescent(self.layers, theta)
                            else:
                                raise ValueError
                        self.optimizer.init_params(self.layers)
                        self.optimizer.update(self.it+1, self.learning_rate)
                    loss = sum(cost) / len(cost)
                    perplexity = math.exp(loss)
                    out_bar.set_postfix(perplexity=perplexity, loss=loss)
                    if (self.it + 1) % save_epoch == 0:
                        self.save_model(path, self.it)
                        print('\n -', self.predict_rnn(prefix, pred_len))
        except KeyboardInterrupt:
            c = input('请输入(Y)保存模型以便继续训练,(C) 继续执行 :')
            if c == 'Y' or c == 'y':
                self.save_model(path, self.it)
                print('已经中断训练。\n再次执行程序，继续从当前开始执行。')
            elif c == 'C' or c == 'c':
                is_continue = True
            else:
                print('结束执行')
        if is_continue:
            start_it = self.it
            is_continue = False
            goto.point

    def predict_rnn(self, prefix, num_chars):
        state = p.zeros([1, self.hidden_unit])
        if len(prefix) > 1:
            output = [self.char_to_ix[prefix[0]]]
        else:
            output = []
        for t in range(num_chars + len(prefix) - 1):
            if t == 0:
                X = p.zeros([1, 1], dtype=int)
            else:
                X = p.array([[output[-1]]]).reshape(1, -1)
            X = self.X_layer.forward(X)
            state = self.rnn_layer.forward(X, state).reshape(1, -1)
            Y = self.out_layer.forward(state)

            if t < len(prefix) - 1:
                output.append(self.char_to_ix[prefix[t + 1]])
            else:
                output.append(int(Y[0].argmax(axis=-1)))

        return ''.join([self.ix_to_char[i] for i in output])

    # 保存模型参数
    def save_model(self, path, start_it):
        for layer in self.layers:
            layer.save_params(path)
        with open(path + os.sep + 'caches.obj', 'wb') as f:
            pickle.dump(start_it, f)
            pickle.dump(self.optimizeMode, f)
        self.optimizer.save_parameters(path)

    # 加载模型参数
    def load_model(self, path):
        for layer in self.layers:
            layer.load_params(path)
        with open(path + os.sep + 'caches.obj', 'rb') as f:
            start_it = pickle.load(f)
            self.optimizeMode = pickle.load(f)
            if self.optimizeMode == 'Adam':
                self.optimizer = Optimize.Adam(self.layers)
            elif self.optimizeMode == 'Momentum':
                self.optimizer = Optimize.Momentum(self.layers)
            elif self.optimizeMode == 'BGD':
                self.optimizer = Optimize.BatchGradientDescent(self.layers)
            else:
                raise ValueError

        self.optimizer.load_parameters(path)
        return start_it
