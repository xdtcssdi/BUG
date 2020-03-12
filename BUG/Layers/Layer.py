import math
import os
import pickle

import numpy

from BUG.Layers.Normalization import BatchNormal
from BUG.Layers.im2col import im2col_indices, col2im_indices_cpu, col2im_indices_gpu
from BUG.function.Activation import ac_get_grad, ac_get
from BUG.load_package import p


def save_struct_params(path, dic):
    with open(path, 'wb') as f:
        pickle.dump(dic, f)


def load_struct_params(path):
    with open(path, 'rb') as f:
        dic = pickle.load(f)
    return dic


class Layer(object):

    def __init__(self, unit_number=0, activation=None):
        self.unit_number = unit_number
        self.activation = activation
        self.parameters = {}
        self.gradients = {}
        self.pre_layer = None
        self.next_layer = None
        self.dZ = None
        self.isFirst = False
        self.isLast = False
        self.batch_normal = None
        self.x = None
        self.Z = None
        self.name = 'layer'

    def init_params(self, nx):
        """
        根据输入的矩阵，初始化参数
        :param nx: 输入矩阵
        :return: None
        """
        raise NotImplementedError

    def forward(self, A_pre, Y=None, mode='train'):
        """
        前向传播
        :param A_pre: 前一层的激活值
        :param mode: 前向传播模式
        :return: 当前层的激活值
        """
        raise NotImplementedError

    def backward(self, pre_grad):
        """
        反向传播
        :param pre_grad: 损失值对当前激活值的导数
        :return: 损失值对前一层激活值的导数
        """
        raise NotImplementedError

    def save_params(self, path, filename):
        """
        保存当前类的参数
        :param path: 路径格式为'xxx/'
        :param filename: 文件名
        :return: None
        """
        raise NotImplementedError

    def load_params(self, path, filename):
        """
        加载npz文件中的参数
        :param path: 路径格式为'xxx/'
        :param filename: 文件名
        :return: None
        """
        raise NotImplementedError


class Convolution(Layer):
    count = 0

    def __init__(self, filter_count, filter_shape,
                 stride=1, padding=0, activation='relu', batchNormal=False):
        """
        :param filter_count: 卷积核数量
        :param filter_shape: 卷积核形状
        :param stride: 步长
        :param padding: pad
        :param activation: 激活函数名字
        :param batchNormal: 是否归一化输出
        """
        super(Convolution, self).__init__(activation=activation)
        Convolution.count += 1
        self.name = 'Convolution_' + str(Convolution.count)
        self.filter_count = filter_count  # 卷积核数量
        self.filter_shape = filter_shape  # 卷积核形状
        self.stride = stride  # 步长
        self.padding = padding  # pad
        self.batchNormal = batchNormal
        self.batch_normal = BatchNormal() if batchNormal else None
        self.args = {'filter_count': filter_count, 'filter_shape': filter_shape,
                     'stride': stride, 'padding': padding, 'activation': activation,
                     'batchNormal': batchNormal}

    def init_params(self, A_pre):  # pre_nc 前一个通道数
        if 'W' not in self.parameters:
            pre_nc = A_pre.shape[1]
            W_shape = (self.filter_count, pre_nc, self.filter_shape[0], self.filter_shape[1])
            n_l = self.filter_shape[0] * self.filter_shape[1] * self.filter_count
            if self.activation == 'relu':  # 'kaiming'
                self.parameters['W'] = p.random.normal(loc=0.0, scale=math.sqrt(2. / n_l), size=W_shape)
            elif self.activation == 'leak_relu':  # 'kaiming'
                self.parameters['W'] = p.random.normal(loc=0.0, scale=math.sqrt(2. / (1.0001 * n_l)), size=W_shape)
            else:
                n_x, d_x, h_x, w_x = A_pre.shape  # 'xavier'
                self.parameters['W'] = p.random.normal(loc=0.0, scale=math.sqrt(2. / (pre_nc + d_x)), size=W_shape)

            self.gradients['W'] = p.zeros_like(self.parameters['W'])
        if 'b' not in self.parameters:
            self.parameters['b'] = p.zeros(self.filter_count)
            self.gradients['b'] = p.zeros_like(self.parameters['W'])

    def save_params(self, path, filename):
        if not os.path.exists(path):
            os.mkdir(path)
        save_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj', self.args)
        p.savez_compressed(path + os.sep + self.name + '_' + filename, W=self.parameters['W'], b=self.parameters['b'])
        if self.batch_normal:
            self.batch_normal.save_params(path + os.sep + self.name + '_' + filename + '_batch_normal')
        return self.name

    def load_params(self, path, filename):
        params = p.load(path + os.sep + self.name + '_' + filename + '.npz')
        dic = load_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj')
        self.filter_count = dic['filter_count']
        self.filter_shape = dic['filter_shape']
        self.stride = dic['stride']
        self.padding = dic['padding']
        self.activation = dic['activation']
        self.parameters['W'] = params['W']
        self.parameters['b'] = params['b']
        if dic['batchNormal']:
            self.batch_normal = BatchNormal()
            self.batch_normal.load_params(path + os.sep + self.name + '_' + filename + '_batch_normal.npz')

    # 没问题
    def forward(self, A_pre, Y=None, mode='train'):
        self.init_params(A_pre)
        self.x = A_pre
        output_data = self.conv(A_pre, self.parameters['W'], self.parameters['b'], self.stride, self.padding)
        self.Z = self.batch_normal.forward(output_data, mode) if self.batch_normal else output_data
        return ac_get(self.Z, self.activation)

    # 没问题
    def backward(self, dout):
        dZ = ac_get_grad(dout, self.Z, self.activation)
        if self.batch_normal:
            dZ = self.batch_normal.backward(dZ)
        self.gradients['b'] = p.sum(dZ, axis=(0, 2, 3))
        num_filters, _, filter_height, filter_width = self.parameters['W'].shape
        dout_reshaped = dZ.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        self.gradients['W'] = dout_reshaped.dot(self.X_col.T).reshape(self.parameters['W'].shape)
        dx_cols = self.parameters['W'].reshape(num_filters, -1).T.dot(dout_reshaped)
        if isinstance(dZ, numpy.ndarray):
            dx = col2im_indices_cpu(dx_cols, self.x.shape, filter_height, filter_width, self.padding, self.stride)
        else:
            dx = col2im_indices_gpu(dx_cols, self.x.shape, filter_height, filter_width, self.padding, self.stride)
        del self.x, self.X_col
        return dx

    def conv(self, X, W, b, stride=1, padding=0):
        """
        卷积的计算过程，向前传播的具体计算
        :param X: 前一层的激活值
        :param W: 过滤器
        :param b: 偏差
        :param stride: 步长
        :param padding: pad
        :return:
        """
        n_filters, d_filter, kernel_size, _ = W.shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = (h_x - kernel_size + 2 * padding) // stride + 1
        w_out = (w_x - kernel_size + 2 * padding) // stride + 1
        h_out, w_out = int(h_out), int(w_out)
        self.X_col = im2col_indices(X, kernel_size, kernel_size, padding=padding, stride=stride)
        W_col = W.reshape(n_filters, -1)
        out = (p.dot(W_col, self.X_col).T + b).T
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out


class Dense(Layer):
    count = 0

    def __init__(self, unit_number, activation=None, batchNormal=False, flatten=False, keep_prob=1.):
        super(Dense, self).__init__(unit_number, activation)
        Dense.count += 1
        self.name = 'Dense_' + str(Dense.count)
        self.flatten = flatten
        self.batch_normal = BatchNormal() if batchNormal else None
        self.keep_prob = keep_prob
        self.drop_mask = None
        self.args = {'unit_number': unit_number, 'activation': activation,
                     'batchNormal': batchNormal, 'flatten': flatten, 'keep_prob': keep_prob}

    def forward(self, x, Y=None, mode='train'):
        self.x_shape = x.shape
        if self.flatten:
            x = x.reshape(x.shape[0], -1)
        self.x = x
        self.init_params(x.shape[-1])
        if x.ndim == 3:
            N, T, D = x.shape
            self.Z = x.reshape(N * T, D).dot(self.parameters['W']).reshape(N, T, self.unit_number) + \
                     self.parameters['b']
        else:
            self.Z = p.dot(x, self.parameters['W']) + self.parameters['b']

        if self.batch_normal:
            self.Z = self.batch_normal.forward(self.Z, mode)
        A = ac_get(self.Z, self.activation)
        if self.keep_prob < 1.:
            self.drop_mask = p.random.rand(*self.Z.shape)
            return A * self.drop_mask / self.keep_prob
        else:
            return A

    def backward(self, dout):
        if self.keep_prob < 1.:
            dout = dout * self.drop_mask / self.keep_prob
        dout = ac_get_grad(dout, self.Z, self.activation)
        if self.batch_normal:
            dout = self.batch_normal.backward(dout)
        if self.x.ndim == 3:
            N, T, D = self.x.shape
            dx = dout.reshape(N * T, self.unit_number).dot(self.parameters['W'].T).reshape(N, T, D)
            self.gradients['W'] = 1. / N * dout.reshape(N * T, self.unit_number).T.dot(self.x.reshape(N * T, D)).T
            self.gradients['b'] = 1. / N * dout.sum(axis=(0, 1))
        else:
            N, D = self.x.shape
            dx = p.dot(dout, self.parameters['W'].T)
            self.gradients['W'] = 1. / N * p.dot(self.x.T, dout)
            self.gradients['b'] = 1. / N * p.sum(dout, axis=0)
        if self.flatten:
            dx = dx.reshape(self.x_shape)  # 还原输入数据的形状（对应张量）
        return dx

    def init_params(self, dim):
        if 'W' not in self.parameters:
            if self.activation == 'relu' or self.activation == 'leak_relu':  # 'Xavier'
                self.parameters['W'] = p.random.uniform(-math.sqrt(6. / (dim + self.unit_number)),
                                                        math.sqrt(6. / (dim + self.unit_number)),
                                                        (dim, self.unit_number))
            elif self.activation == 'tanh' or self.activation == 'sigmoid':
                self.parameters['W'] = p.random.uniform(-1., 1., (dim, self.unit_number)) \
                                       * p.sqrt(6. / (dim + self.unit_number))
            else:  # 'MSRA'
                self.parameters['W'] = p.random.normal(0, math.sqrt(2. / dim), size=(dim, self.unit_number))
            # self.parameters['W'] = p.random.randn(pre_unit, self.unit_number) * 0.01
        if 'b' not in self.parameters:
            self.parameters['b'] = p.zeros(self.unit_number)

    def save_params(self, path, filename):
        save_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj', self.args)
        p.savez_compressed(path + os.sep + self.name + '_' + filename, W=self.parameters['W'], b=self.parameters['b'])
        if self.batch_normal:
            self.batch_normal.save_params(path + os.sep + self.name + '_' + filename + '_batch_normal')
        return self.name

    def load_params(self, path, filename):
        dic = load_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj')
        self.unit_number = dic['unit_number']
        self.activation = dic['activation']
        if dic['batchNormal']:
            self.batch_normal = BatchNormal()
            self.batch_normal.load_params(path + os.sep + self.name + '_' + filename + '_batch_normal.npz')
        r = p.load(path + os.sep + self.name + '_' + filename + '.npz')
        self.parameters['W'] = r['W']
        self.parameters['b'] = r['b']


class Pooling(Layer):
    count = 0

    def __init__(self, filter_shape, paddingMode='same', stride=1, mode='max'):
        super(Pooling, self).__init__()
        Pooling.count += 1
        self.filter_shape = filter_shape
        self.name = 'Pooling_' + str(Pooling.count)
        self.stride = stride
        self.padding = 0 if paddingMode == 'valid' else (filter_shape[0] - 1) // 2
        self.mode = mode
        self.args = {'filter_shape': filter_shape, 'paddingMode': paddingMode,
                     'stride': stride, 'mode': mode}
        assert (self.mode in ['max', 'average'])

    def init_params(self, nx):
        pass

    def forward(self, A_pre, Y=None, mode='train'):
        N, C, H, W = A_pre.shape

        out_height = (H - self.filter_shape[0]) // self.stride + 1
        out_width = (W - self.filter_shape[1]) // self.stride + 1

        x_split = A_pre.reshape(N * C, 1, H, W)
        x_cols = im2col_indices(x_split, self.filter_shape[0], self.filter_shape[1], padding=self.padding,
                                stride=self.stride)
        x_cols_argmax = p.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, p.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
        self.cache = (A_pre, x_cols, x_cols_argmax)
        return out

    def backward(self, dZ):
        x, x_cols, x_cols_argmax = self.cache
        del self.cache
        N, C, H, W = x.shape

        dout_reshaped = dZ.transpose(2, 3, 0, 1).flatten()
        dx_cols = p.zeros_like(x_cols)
        dx_cols[x_cols_argmax, p.arange(dx_cols.shape[1])] = dout_reshaped
        if isinstance(x, numpy.ndarray):
            dx = col2im_indices_cpu(dx_cols, (N * C, 1, H, W), self.filter_shape[0], self.filter_shape[1],
                                    padding=self.padding, stride=self.stride)
        else:
            dx = col2im_indices_gpu(dx_cols, (N * C, 1, H, W), self.filter_shape[0], self.filter_shape[1],
                                    padding=self.padding, stride=self.stride)
        dx = dx.reshape(x.shape)
        return dx

    def save_params(self, path, filename):
        save_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj', self.args)
        return self.name

    def load_params(self, path, filename):
        r = load_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj')
        self.filter_shape = r['filter_shape']
        self.padding = 0 if r['paddingMode'] == 'valid' else (self.filter_shape[0] - 1) // 2
        self.stride = r['stride']
        self.mode = r['mode']


class SimpleRNN(Layer):

    def save_params(self, path, filename):
        pass

    def load_params(self, path, filename):
        pass

    def __init__(self, char_to_ix, n_a=50, learning_rate=0.01):
        super(SimpleRNN, self).__init__()
        self.n_a = n_a
        self.char_to_ix = char_to_ix
        self.learning_rate = learning_rate
        self.a0 = None
        self.a = None

    def init_params(self, n_x):
        if 'Waa' not in self.parameters:
            self.parameters['Waa'] = p.random.randn(self.n_a, self.n_a) * 0.01
            self.parameters['Wax'] = p.random.randn(self.n_a, n_x) * 0.01
            self.parameters['b'] = p.zeros(self.n_a)

    def rnn_step_forward(self, x, a_prev):
        a = p.dot(a_prev, self.parameters['Waa']) + p.dot(x, self.parameters['Wax']) + self.parameters['b']
        a_next = p.tanh(a)
        return a_next

    def rnn_step_backward(self, da_next, x, a_prev, a_next):
        da = da_next * (1 - a_next * a_next)
        dx = p.dot(da, self.parameters['Wax'].T)
        da_prev = p.dot(da, self.parameters['Waa'].T)
        dWx = p.dot(x.T, da)
        dWa = p.dot(a_prev.T, da)
        db = p.sum(da, axis=0)
        return dx, da_prev, dWx, dWa, db

    def forward(self, x, a0=None, mode='train'):
        self.x = x
        m, T_x, self.n_x = x.shape
        self.init_params(self.n_x)
        self.a = p.zeros([m, T_x, self.n_a])
        a_prev = a0 if a0 is not None else p.zeros([m, self.n_a])
        self.a0 = a_prev
        for t in range(T_x):
            xt = x[:, t, :]
            a_next = self.rnn_step_forward(xt, a_prev)
            a_prev = a_next
            self.a[:, t, :] = a_next
        return self.a

    def backward(self, da):

        m, T_x, n_a = da.shape
        n_x = self.parameters['b'].shape[0]

        a_next = self.a[:, T_x - 1, :]

        da_prev = p.zeros([m, n_a])
        dx = p.zeros([m, T_x, n_x])
        self.gradients['Waa'] = p.zeros_like(self.parameters['Waa'])
        self.gradients['Wax'] = p.zeros_like(self.parameters['Wax'])
        self.gradients['b'] = p.zeros_like(self.parameters['b'])
        for t in range(T_x):
            t = T_x - 1 - t
            xt = self.x[:, t, :]
            a_prev = self.a0 if t == 0 else self.a[:, t - 1, :]

            da_next = da[:, t, :] + da_prev
            dx[:, t, :], da_prev, dWxt, dWat, dbt = self.rnn_step_backward(da_next, xt, a_prev, a_next)
            a_next = a_prev
            self.gradients['Wax'] += dWxt
            self.gradients['Waa'] += dWat
            self.gradients['b'] += dbt

        da0 = da_prev
        return dx, da0


class LSTM(Layer):
    def __init__(self, word_to_idx, point, n_a=50):
        super(LSTM, self).__init__()
        self.cache = {}
        self.name = 'LSTM'
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.start_code, self.end_code, self.null_code = point
        self.n_a = n_a
        self.args = {'word_to_idx': word_to_idx, 'point': point, 'n_a': n_a}

    def init_params(self, n_x):
        if 'Wx' not in self.parameters:
            self.parameters['Wx'] = p.random.randn(n_x, 4 * self.n_a) / p.sqrt(n_x)
            self.parameters['Wa'] = p.random.randn(self.n_a, 4 * self.n_a) / p.sqrt(self.n_a)
            self.parameters['b'] = p.zeros(4 * self.n_a)

    def forward(self, x, a0=None, mode='train'):
        """
        :param x: 二维矩阵 [m, n_x]
        :param a0:
        :param mode:
        :return:
        """
        self.init_params(x.shape[-1])
        m, time_steps, n_x = x.shape
        n_a = int(self.parameters['b'].shape[0] / 4)
        a = p.zeros([m, time_steps, n_a])
        a_prev = a0 if a0 is not None else p.zeros([m, n_a])
        c_prev = p.zeros([m, n_a])

        for t in range(time_steps):
            xt = x[:, t, :]
            a_next, c_next, self.cache[t] = self.lstm_step_forward(xt, a_prev, c_prev)
            a_prev = a_next
            c_prev = c_next
            a[:, t, :] = a_prev
        return a

    def backward(self, dout):
        m, time_steps, n_a = dout.shape
        z_i, z_f, z_o, z_g, z_t, prev_c, prev_a, x = self.cache[time_steps - 1]
        n_x = x.shape[1]

        da_prev = p.zeros((m, n_a))
        dc_prev = p.zeros((m, n_a))
        dx = p.zeros((m, time_steps, n_x))
        dWx = p.zeros((n_x, 4 * n_a))
        dWa = p.zeros((n_a, 4 * n_a))
        db = p.zeros((4 * n_a,))

        for t in range(time_steps):
            t = time_steps - 1 - t
            da_next = dout[:, t, :] + da_prev
            dc_next = dc_prev
            dx[:, t, :], da_prev, dc_prev, dWxt, dWat, dbt = self.lstm_step_backward(da_next, dc_next, self.cache[t])
            dWx, dWa, db = dWx + dWxt, dWa + dWat, db + dbt

        da0 = da_prev
        self.gradients['a'] = da0
        self.gradients['x'] = dx
        self.gradients['Wx'] = dWx
        self.gradients['Wa'] = dWa
        self.gradients['b'] = db
        return dx, da0

    def lstm_step_forward(self, x, a_prev, c_prev):
        n_a = self.parameters['Wa'].shape[0]
        a = x.dot(self.parameters['Wx']) + a_prev.dot(self.parameters['Wa']) + self.parameters['b']

        z_i = ac_get(a[:, :n_a], 'sigmoid')
        z_f = ac_get(a[:, n_a:2 * n_a], 'sigmoid')
        z_o = ac_get(a[:, 2 * n_a:3 * n_a], 'sigmoid')
        z_g = p.tanh(a[:, 3 * n_a:])

        c_next = z_f * c_prev + z_i * z_g
        z_t = p.tanh(c_next)
        a_next = z_o * z_t
        cache = (z_i, z_f, z_o, z_g, z_t, c_prev, a_prev, x)
        return a_next, c_next, cache

    def lstm_step_backward(self, da_next, dc_next, cache):

        z_i, z_f, z_o, z_g, z_t, c_prev, a_prev, x = cache

        dz_o = z_t * da_next
        dc_t = z_o * (1 - z_t * z_t) * da_next + dc_next
        dz_f = c_prev * dc_t
        dz_i = z_g * dc_t
        dc_prev = z_f * dc_t
        dz_g = z_i * dc_t

        da_i = (1 - z_i) * z_i * dz_i
        da_f = (1 - z_f) * z_f * dz_f
        da_o = (1 - z_o) * z_o * dz_o
        da_g = (1 - z_g * z_g) * dz_g
        da = p.hstack((da_i, da_f, da_o, da_g))

        dWx = x.T.dot(da)
        dWa = a_prev.T.dot(da)

        db = p.sum(da, axis=0)
        dx = da.dot(self.parameters['Wx'].T)
        da_prev = da.dot(self.parameters['Wa'].T)

        return dx, da_prev, dc_prev, dWx, dWa, db

    def save_params(self, path, filename):
        save_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj', self.args)
        p.savez_compressed(path + os.sep + self.name + '_' + filename, Wx=self.parameters['Wx'],
                           Wa=self.parameters['Wa'], b=self.parameters['b'])
        return self.name

    def load_params(self, path, filename):
        dic = load_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj')
        self.word_to_idx = dic['word_to_idx']
        self.n_a = dic['n_a']

        r = p.load(path + os.sep + self.name + '_' + filename + '.npz')
        self.parameters['Wx'] = r['Wx']
        self.parameters['Wa'] = r['Wa']
        self.parameters['b'] = r['b']


class Embedding(Layer):
    def __init__(self, vocab_size, word_dim):
        super(Embedding, self).__init__()
        self.vocab_size = vocab_size
        self.word_dim = word_dim
        self.init_params([self.vocab_size, self.word_dim])
        self.name = 'Embedding'
        self.args = {'vocab_size': vocab_size, 'word_dim': word_dim}

    def init_params(self, dim):
        self.parameters['W'] = p.random.randn(*dim) / 100

    def forward(self, x, Y=None, mode='train'):
        self.x = x
        N, T = x.shape
        out = p.zeros((N, T, self.word_dim))

        for i in range(N):
            for j in range(T):
                out[i, j] = self.parameters['W'][x[i, j]]
        return out

    def backward(self, dout):
        dW = p.zeros_like(self.parameters['W'])
        if isinstance(dW, numpy.ndarray):
            p.add.at(dW, self.x, dout)
        else:
            p.scatter_add(dW, self.x, dout)
        self.gradients['W'] = dW

    def save_params(self, path, filename):
        if not os.path.exists(path):
            os.mkdir(path)
        save_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj', self.args)
        p.savez_compressed(path + os.sep + self.name + '_' + filename, W=self.parameters['W'])
        return self.name

    def load_params(self, path, filename):
        params = p.load(path + os.sep + self.name + '_' + filename + '.npz')
        dic = load_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj')
        self.vocab_size = dic['vocab_size']
        self.word_dim = dic['word_dim']
        self.parameters['W'] = params['W']


def generate_layer(str, arg):
    if str == 'Dense':
        return Dense(**arg)
    elif str == 'Convolution':
        return Convolution(**arg)
    elif str == 'Pooling':
        return Pooling(**arg)
    elif str == 'SimpleRNN':
        return SimpleRNN(**arg)
    elif str == 'LSTM':
        return LSTM(**arg)
    elif str == 'Embedding':
        return Embedding(**arg)
    else:
        raise ValueError
