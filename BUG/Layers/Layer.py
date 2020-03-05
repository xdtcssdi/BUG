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

    def __init__(self, unit_number=0, activation="relu"):
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
        self.A_pre = None
        self.Z = None
        self.name = 'layer'

    def init_params(self, nx):
        '''
        根据输入的矩阵，初始化参数
        :param nx: 输入矩阵
        :return: None
        '''
        raise NotImplementedError

    def forward(self, A_pre, mode='train'):
        '''
        前向传播
        :param A_pre: 前一层的激活值
        :param Y: 训练集
        :param mode: 前向传播模式
        :return: 当前层的激活值
        '''
        raise NotImplementedError

    def backward(self, pre_grad):
        '''
        反向传播
        :param pre_grad: 损失值对当前激活值的导数
        :return: 损失值对前一层激活值的导数
        '''
        raise NotImplementedError

    def save_params(self, path, filename):
        '''
        保存当前类的参数
        :param path: 路径格式为'xxx/'
        :param filename: 文件名
        :return: None
        '''
        raise NotImplementedError

    def load_params(self, path, filename):
        '''
        加载npz文件中的参数
        :param path: 路径格式为'xxx/'
        :param filename: 文件名
        :return: None
        '''
        raise NotImplementedError


class Convolution(Layer):
    count = 0

    def __init__(self, filter_count, filter_shape,
                 stride=1, padding=0, activation='relu', batchNormal=False):
        '''
        :param filter_count: 卷积核数量
        :param filter_shape: 卷积核形状
        :param stride: 步长
        :param padding: pad
        :param activation: 激活函数名字
        :param batchNormal: 是否归一化输出
        '''
        super(Convolution, self).__init__(activation=activation)
        Convolution.count += 1
        self.name = 'Convolution_' + str(Convolution.count)
        self.filter_count = filter_count  # 卷积核数量
        self.filter_shape = filter_shape  # 卷积核形状
        self.stride = stride  # 步长
        self.padding = padding  # pad
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

            self.gradient['dW'] = p.zeros_like(self.parameters['W'])
        if 'b' not in self.parameters:
            self.parameters['b'] = p.zeros(self.filter_count)
            self.gradient['db'] = p.zeros_like(self.parameters['W'])

    def save_params(self, path, filename):
        if not os.path.exists(path):
            os.mkdir(path)
        save_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj', self.args)
        p.savez_compressed(path + os.sep + self.name + '_' + filename, W=self.parameters['W'], b=self.parameters['b'])
        if self.batch_normal:
            self.batch_normal.save_params(path + os.sep + self.name + '_' + filename + '_batch_normal')

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
        self.A_pre = A_pre
        output_data = self.conv(A_pre, self.parameters['W'], self.parameters['b'], self.stride, self.padding)
        self.Z = self.batch_normal.forward(output_data, mode) if self.batch_normal else output_data
        return ac_get(self.Z, self.activation)

    # 没问题
    def backward(self, dout):
        dZ = ac_get_grad(dout, self.Z, self.activation)
        if self.batch_normal:
            dZ = self.batch_normal.backward(dZ)
        self.gradients['db'] = p.sum(dZ, axis=(0, 2, 3))
        num_filters, _, filter_height, filter_width = self.parameters['W'].shape
        dout_reshaped = dZ.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        self.gradients['dW'] = dout_reshaped.dot(self.X_col.T).reshape(self.parameters['W'].shape)
        dx_cols = self.parameters['W'].reshape(num_filters, -1).T.dot(dout_reshaped)
        if isinstance(dZ, numpy.ndarray):
            dx = col2im_indices_cpu(dx_cols, self.A_pre.shape, filter_height, filter_width, self.padding, self.stride)
        else:
            dx = col2im_indices_gpu(dx_cols, self.A_pre.shape, filter_height, filter_width, self.padding, self.stride)
        del self.A_pre, self.X_col
        return dx

    def conv(self, X, W, b, stride=1, padding=0):
        '''
        卷积的计算过程，向前传播的具体计算
        :param X: 前一层的激活值
        :param W: 过滤器
        :param b: 偏差
        :param stride: 步长
        :param padding: pad
        :return:
        '''
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


class Core(Layer):
    count = 0

    def __init__(self, unit_number, activation="relu", batchNormal=False):
        super(Core, self).__init__(unit_number, activation)
        Core.count += 1
        self.name = 'Core_' + str(Core.count)
        self.batch_normal = BatchNormal() if batchNormal else None
        self.args = {'unit_number': unit_number, 'activation': activation, 'batchNormal': batchNormal}

    def forward(self, x, Y=None, mode='train'):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.init_params(x)
        self.x = x
        self.Z = p.dot(self.x, self.parameters['W']) + self.parameters['b']
        if self.batch_normal:
            self.Z = self.batch_normal.forward(self.Z)
        return ac_get(self.Z, self.activation)

    def backward(self, dout):
        dout = ac_get_grad(dout, self.Z, self.activation)
        if self.batch_normal:
            dout = self.batch_normal.backward(dout)
        dx = p.dot(dout, self.parameters['W'].T)
        self.gradients['dW'] = p.dot(self.x.T, dout)
        self.gradients['db'] = p.sum(dout, axis=0)
        dx = dx.reshape(self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx

    def init_params(self, A_pre):
        pre_unit = A_pre.shape[1]
        if 'W' not in self.parameters:
            if self.activation == 'relu' or self.activation == 'leak_relu':  # 'Xavier'
                self.parameters['W'] = p.random.uniform(-math.sqrt(6. / (pre_unit + self.unit_number)),
                                                        math.sqrt(6. / (pre_unit + self.unit_number)),
                                                        (pre_unit, self.unit_number))
            elif self.activation == 'tanh' or self.activation == 'sigmoid':
                self.parameters['W'] = p.random.uniform(-1., 1., (pre_unit, self.unit_number)) \
                                       * p.sqrt(6. / (pre_unit + self.unit_number))
            else:  # 'MSRA'
                self.parameters['W'] = p.random.normal(0, math.sqrt(2. / pre_unit), size=(pre_unit, self.unit_number))
            # self.W = p.random.randn(pre_unit, self.unit_number) * 0.01
        if 'b' not in self.parameters:
            self.parameters['b'] = p.zeros((1, self.unit_number))

    def save_params(self, path, filename):
        save_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj', self.args)
        p.savez_compressed(path + os.sep + self.name + '_' + filename, W=self.parameters['W'], b=self.parameters['b'])
        if self.batch_normal:
            self.batch_normal.save_params(path + os.sep + self.name + '_' + filename + '_batch_normal')

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
        self.args = {'filter_shape': filter_shape, 'padding': self.padding,
                     'stride': stride, 'mode': mode}
        assert (self.mode in ['max', 'average'])

    def save_params(self, path, filename):
        save_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj', self.args)

    def load_params(self, path, filename):
        r = load_struct_params(path + os.sep + self.name + '_' + filename + '_struct.obj')
        self.filter_shape = r['filter_shape']
        self.padding = r['padding']
        self.stride = r['stride']
        self.mode = r['mode']

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


class SimpleRNN(Layer):

    def save_params(self, path, filename):
        pass

    def load_params(self, path, filename):
        pass

    def __init__(self, n_x, n_y, ix_to_char, char_to_ix, n_a=50, learning_rate=0.01, activation='softmax'):
        super(SimpleRNN, self).__init__(activation=activation)
        self.n_a = n_a
        self.ix_to_char = ix_to_char
        self.char_to_ix = char_to_ix
        self.init_params((n_x, n_y))
        self.learning_rate = learning_rate

    def init_params(self, n_x_y):
        self.n_x, self.n_y = n_x_y
        self.parameters['Waa'] = p.random.randn(self.n_a, self.n_a) * 0.01
        self.parameters['Wax'] = p.random.randn(self.n_a, self.n_x) * 0.01
        self.parameters['Wya'] = p.random.randn(self.n_y, self.n_a) * 0.01
        self.gradients['dWaa'] = p.zeros_like(self.parameters['Waa'])
        self.gradients['dWax'] = p.zeros_like(self.parameters['Wax'])
        self.gradients['dWya'] = p.zeros_like(self.parameters['Wya'])
        self.parameters['b'] = p.zeros((self.n_a, 1))
        self.parameters['by'] = p.zeros((self.n_y, 1))
        self.gradients['db'] = p.zeros_like(self.parameters['b'])
        self.gradients['dby'] = p.zeros_like(self.parameters['by'])

    def softmax(self, x):
        e_x = p.exp(x - p.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, X, a0=None, mode='train'):
        '''
        :param X: shape = (batch_size, time_steps, vocab_size)
        :param a0:
        :param mode:
        :return:
        '''
        self.X = X
        self.a0 = a0 if a0 is not None else p.zeros([self.n_a, X.shape[0]])
        m, T_x, n_x = X.shape
        self.y_hat = p.zeros([X.shape[0], T_x, self.n_y])
        self.a = p.zeros([self.n_a, T_x, X.shape[0]])
        if a0 is not None:
            self.a[:, 0, :] = a0

        for t in range(T_x - 1):
            at, self.y_hat[:, t, :] = self.rnn_step_forward(self.a[:, t, :], X[:, t, :])
            self.a[:, t + 1, :] = at
        self.at, self.y_hat[:, T_x - 1, :] = self.rnn_step_forward(self.a[:, T_x - 1, :], X[:, T_x - 1, :])
        return self.at, self.y_hat

    def backward(self, dout):
        self.gradients['dWaa'] = p.zeros_like(self.parameters['Waa'])
        self.gradients['dWax'] = p.zeros_like(self.parameters['Wax'])
        self.gradients['dWya'] = p.zeros_like(self.parameters['Wya'])
        self.gradients['db'] = p.zeros_like(self.parameters['b'])
        self.gradients['dby'] = p.zeros_like(self.parameters['by'])
        da_next = self.at
        for t in reversed(range(self.X.shape[1])):
            da_next = self.rnn_step_backward(dout[:, t, :], self.X[:, t, :], self.a[:, t, :],
                                             self.a[:, t - 1, :] if t != 0 else self.a0, da_next)
        return da_next

    def rnn_step_forward(self, a_prev, x):
        a_next = p.tanh(
            p.dot(self.parameters['Wax'], x.T) + p.dot(self.parameters['Waa'], a_prev) + self.parameters['b'])
        y_hat = self.softmax(p.dot(self.parameters['Wya'], a_next) + self.parameters['by'])
        return a_next, y_hat.T

    def rnn_step_backward(self, dout, x, a, a_prev, da_next):
        self.gradients['dWya'] += p.dot(a, dout).T
        self.gradients['dby'] += p.mean(dout.T, axis=1, keepdims=True)
        da = p.dot(dout, self.parameters['Wya']).T + da_next
        daraw = (1 - a * a) * da
        self.gradients['db'] += p.mean(daraw, axis=1, keepdims=True)
        self.gradients['dWax'] += p.dot(daraw, x)
        self.gradients['dWaa'] += p.dot(daraw, a_prev.T)
        da_next = p.dot(self.parameters['Waa'].T, daraw)
        return da_next


class LSTM(Layer):

    def __init__(self, n_x, n_y, ix_to_char, char_to_ix, n_a=50):
        super(LSTM, self).__init__()
        self.n_a = n_a
        self.ix_to_char = ix_to_char
        self.char_to_ix = char_to_ix
        self.gradients = {}
        self.parameters = {}
        self.init_params([n_a, n_x, n_y])

    def init_params(self, n_a_x_y):
        n_a, n_x, self.n_y = n_a_x_y
        self.parameters['Wf'] = p.random.randn(n_a, n_a + n_x)
        self.parameters['bf'] = p.random.randn(n_a, 1)
        self.parameters['Wi'] = p.random.randn(n_a, n_a + n_x)
        self.parameters['bi'] = p.random.randn(n_a, 1)
        self.parameters['Wo'] = p.random.randn(n_a, n_a + n_x)
        self.parameters['bo'] = p.random.randn(n_a, 1)
        self.parameters['Wc'] = p.random.randn(n_a, n_a + n_x)
        self.parameters['bc'] = p.random.randn(n_a, 1)
        self.parameters['Wy'] = p.random.randn(self.n_y, n_a)
        self.parameters['by'] = p.random.randn(self.n_y, 1)

    def softmax(self, x):
        e_x = p.exp(x - p.max(x))
        return e_x / e_x.sum(axis=0)

    def forward(self, X, a0=None, mode='train'):
        self.x = X
        n_x, m, T_x = X.shape
        n_y, n_a = self.parameters['Wy'].shape
        self.a = p.zeros((n_a, m, T_x))
        c = p.zeros((n_a, m, T_x))
        y = p.zeros((n_y, m, T_x))
        a_next = a0
        c_next = p.zeros_like(a_next)
        for t in range(T_x):
            a_next, c_next, yt_pred = self.lstm_step_forward(a_next, X[..., t], c_next, )
            self.a[..., t] = a_next
            y[..., t] = yt_pred
            c[..., t] = c_next

        return a_next, y, c

    def backward(self, dout):
        n_x, m, T_x = self.x.shape
        self.gradients['dx'] = p.zeros([n_x, m, T_x])
        self.gradients['dWf'] = p.zeros([self.n_a, self.n_a + n_x])
        self.gradients['dWi'] = p.zeros([self.n_a, self.n_a + n_x])
        self.gradients['dWc'] = p.zeros([self.n_a, self.n_a + n_x])
        self.gradients['dWo'] = p.zeros([self.n_a, self.n_a + n_x])
        self.gradients['dWy'] = p.zeros([self.n_y, self.n_a])
        self.gradients['dby'] = p.zeros([self.n_y, 1])
        self.gradients['dbf'] = p.zeros([self.n_a, 1])
        self.gradients['dbi'] = p.zeros([self.n_a, 1])
        self.gradients['dbc'] = p.zeros([self.n_a, 1])
        self.gradients['dbo'] = p.zeros([self.n_a, 1])

        da_prev = p.zeros([self.n_a, m])
        dc_prev = p.zeros([self.n_a, m])
        for t in reversed(range(T_x)):
            gradients = self.lstm_step_backward(dout[:, :, t], self.x[..., t], self.a[..., t], da_prev, dc_prev)
            self.gradients['dx'][:, :, t] = gradients['dx']
            self.gradients['dWf'] += gradients['dWf']
            self.gradients['dWi'] += gradients['dWi']
            self.gradients['dWc'] += gradients['dWc']
            self.gradients['dWo'] += gradients['dWo']
            self.gradients['dbf'] += gradients['dbf']
            self.gradients['dbi'] += gradients['dbi']
            self.gradients['dbc'] += gradients['dbc']
            self.gradients['dbo'] += gradients['dbo']
            self.gradients['dWy'] += gradients['dWy']
            self.gradients['dby'] += gradients['dby']
            da_prev = gradients['da_prev']
            dc_prev = gradients['dc_prev']
        # da0 = gradients['da_prev']

    def save_params(self, path, filename):
        pass

    def load_params(self, path, filename):
        pass

    def lstm_step_forward(self, a_prev, x, c_prev):
        # print(a_prev.shape, x.shape)
        merge = p.concatenate([a_prev, x], axis=0)
        f = ac_get(p.dot(self.parameters['Wf'], merge) + self.parameters['bf'], 'sigmoid')
        i = ac_get(p.dot(self.parameters['Wi'], merge) + self.parameters['bi'], 'sigmoid')
        c_hat = p.tanh(p.dot(self.parameters['Wc'], merge) + self.parameters['bc'])
        c = f * c_prev + i * c_hat
        o = ac_get(p.dot(self.parameters['Wo'], merge) + self.parameters['bo'], 'sigmoid')
        a = o * p.tanh(c)
        y = ac_get(p.dot(self.parameters['Wy'], a) + self.parameters['by'], 'softmax')
        self.caches = (f, i, c_hat, c_prev, c, o, a_prev, a)
        return a, c, y

    def lstm_step_backward(self, dout, xt, at, da_next, dc_next):
        f, i, c_hat, c_prev, c_next, o, a_prev, a_next = self.caches
        n_a, m = a_next.shape
        do = da_next * p.tanh(c_next) * o * (1 - o)
        dc_hat = (dc_next * i + o * (1 - p.square(p.tanh(c_next))) * i * da_next) * (1 - p.square(c_hat))
        di = (dc_next * c_hat + o * (1 - p.square(p.tanh(c_next))) * c_hat * da_next) * i * (1 - i)
        df = (dc_next * c_prev + o * (1 - p.square(p.tanh(c_next))) * c_prev * da_next) * f * (1 - f)
        concat = p.concatenate((a_prev, xt), axis=0).T
        gradient = {}
        gradient['dWy'] = p.dot(dout, at.T)
        gradient['dby'] = p.mean(dout, axis=1, keepdims=True)
        gradient['dWf'] = p.dot(df, concat)
        gradient['dWi'] = p.dot(di, concat)
        gradient['dWc'] = p.dot(dc_hat, concat)
        gradient['dWo'] = p.dot(do, concat)
        gradient['dbf'] = p.sum(df, axis=1, keepdims=True)
        gradient['dbi'] = p.sum(di, axis=1, keepdims=True)
        gradient['dbc'] = p.sum(dc_hat, axis=1, keepdims=True)
        gradient['dbo'] = p.sum(do, axis=1, keepdims=True)
        gradient['da_prev'] = p.dot(self.parameters['Wf'][:, :n_a].T, df) + p.dot(self.parameters['Wc'][:, :n_a].T,
                                                                                  dc_hat) + \
                              p.dot(self.parameters['Wi'][:, :n_a].T, di) + p.dot(self.parameters['Wo'][:, :n_a].T, do)
        gradient['dc_prev'] = dc_next * f + o * (1 - p.square(p.tanh(c_next))) * f * da_next
        gradient['dx'] = p.dot(self.parameters['Wf'][:, n_a:].T, df) + p.dot(self.parameters['Wc'][:, n_a:].T, dc_hat) + \
                         p.dot(self.parameters['Wi'][:, n_a:].T, di) + p.dot(self.parameters['Wo'][:, n_a:].T, do)

        return gradient
