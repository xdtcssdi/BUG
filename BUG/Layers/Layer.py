import math

import numpy as np

from BUG.Layers.Normalization import BatchNormal
from BUG.Layers.im2col import im2col_indices, col2im_indices
from BUG.function.Activation import ac_get_grad, ac_get


class Layer(object):

    def __init__(self, unit_number=0, activation="relu"):
        self.unit_number = unit_number
        self.activation = activation
        self.pre_layer = None
        self.next_layer = None
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.dZ = None
        self.isFirst = False
        self.isLast = False
        self.batchNormal = None
        self.A_pre = None
        self.Z = None
        self.name = 'layer'

    def init_params(self, nx):
        raise NotImplementedError

    def forward(self, A_pre, mode='train'):
        raise NotImplementedError

    def backward(self, pre_grad):
        raise NotImplementedError


class Convolution(Layer):

    def __init__(self, filter_count, filter_shape,
                 stride=1, padding=0, activation='relu', batchNormal=False):
        super(Convolution, self).__init__(activation=activation)
        self.name = 'Convolution'
        self.filter_count = filter_count  # 卷积核数量
        self.filter_shape = filter_shape  # 卷积核形状
        self.stride = stride  # 步长
        self.padding = padding  # pad
        self.batchNormal = BatchNormal() if batchNormal else None

    def init_params(self, A_pre):  # pre_nc 前一个通道数
        pre_nc = A_pre.shape[1]

        if self.W is None:
            W_shape = (self.filter_count, pre_nc, self.filter_shape[0], self.filter_shape[1])
            n_l = self.filter_shape[0] * self.filter_shape[1] * self.filter_count
            if self.activation == 'relu':  # 'kaiming'
                self.W = np.random.normal(loc=0.0, scale=math.sqrt(2. / n_l), size=W_shape)
            elif self.activation == 'leak_relu':  # 'kaiming'
                self.W = np.random.normal(loc=0.0, scale=math.sqrt(2. / (1.0001 * n_l)), size=W_shape)
            else:
                n_x, d_x, h_x, w_x = A_pre.shape  # 'xavier'
                self.W = np.random.normal(loc=0.0, scale=math.sqrt(2. / (pre_nc + d_x)), size=W_shape)
            self.dW = np.zeros_like(self.W)

        if self.b is None:
            self.b = np.zeros(self.filter_count)
            self.db = np.zeros_like(self.b)

    # 没问题
    def forward(self, A_pre, mode='train'):
        self.init_params(A_pre)
        self.A_pre = A_pre
        output_data = self.conv(A_pre, self.W, self.b, self.stride, self.padding)
        Z = self.batchNormal.forward(output_data, mode) if self.batchNormal else output_data
        return ac_get(Z, self.activation)

    # 没问题
    def backward(self, dZ):
        if self.batchNormal:
            dZ = self.batchNormal.backward(dZ)
        self.db = np.sum(dZ, axis=(0, 2, 3))
        num_filters, _, filter_height, filter_width = self.W.shape
        dout_reshaped = dZ.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        self.dW = dout_reshaped.dot(self.X_col.T).reshape(self.W.shape)
        dx_cols = self.W.reshape(num_filters, -1).T.dot(dout_reshaped)
        dx = col2im_indices(dx_cols, self.A_pre.shape, filter_height, filter_width, self.padding, self.stride)
        dZ = ac_get_grad(dx, self.A_pre, self.activation)
        del self.A_pre, self.X_col
        return dZ

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db

    def conv(self, X, W, b, stride=1, padding=0):  # 卷积的计算过程，向前传播的具体计算
        n_filters, d_filter, kernel_size, _ = W.shape
        n_x, d_x, h_x, w_x = X.shape
        h_out = (h_x - kernel_size + 2 * padding) // stride + 1
        w_out = (w_x - kernel_size + 2 * padding) // stride + 1
        h_out, w_out = int(h_out), int(w_out)
        self.X_col = im2col_indices(X, kernel_size, kernel_size, padding=padding, stride=stride)
        W_col = W.reshape(n_filters, -1)
        out = (np.dot(W_col, self.X_col).T + b).T
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out


class Core(Layer):

    def __init__(self, unit_number, activation="relu", batchNormal=False):
        super(Core, self).__init__(unit_number, activation)
        self.batchNormal = BatchNormal() if batchNormal else None
        self.name = 'Core'

    def forward(self, x, mode='train'):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.init_params(x)
        self.x = x
        self.out = np.dot(self.x, self.W) + self.b
        if self.batchNormal:
            self.out = self.batchNormal.forward(self.out)
        return ac_get(self.out, self.activation)

    def backward(self, dout):
        dout = ac_get_grad(dout, self.out, self.activation)
        if self.batchNormal:
            dout = self.batchNormal.backward(dout)
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx

    def init_params(self, A_pre):
        pre_unit = A_pre.shape[1]
        if self.W is None:
            if self.activation == 'relu' or self.activation == 'leak_relu':  # 'Xavier'
                self.W = np.random.uniform(-math.sqrt(6. / (pre_unit + self.unit_number)),
                                           math.sqrt(6. / (pre_unit + self.unit_number)),
                                           (pre_unit, self.unit_number))
            elif self.activation == 'tanh' or self.activation == 'sigmoid':
                self.W = np.random.uniform(-1., 1., (pre_unit, self.unit_number)) \
                         * np.sqrt(6. / (pre_unit + self.unit_number))
            else:  # 'MSRA'
                self.W = np.random.normal(0, math.sqrt(2. / pre_unit), size=(pre_unit, self.unit_number))
            # self.W = np.random.randn(pre_unit, self.unit_number) * 0.01
        if self.b is None:
            self.b = np.zeros((1, self.unit_number))

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db


class Pooling(Layer):
    def __init__(self, filter_shape, paddingMode='same', stride=1, mode='max'):
        super(Pooling, self).__init__()
        self.filter_shape = filter_shape
        self.name = 'Pooling'
        self.stride = stride
        self.padding = 0 if paddingMode == 'valid' else (filter_shape[0] - 1) // 2
        self.mode = mode
        assert (self.mode in ['max', 'average'])

    def init_params(self, nx):
        pass

    def forward(self, A_pre, mode='train'):
        N, C, H, W = A_pre.shape

        out_height = (H - self.filter_shape[0]) // self.stride + 1
        out_width = (W - self.filter_shape[1]) // self.stride + 1

        x_split = A_pre.reshape(N * C, 1, H, W)
        x_cols = im2col_indices(x_split, self.filter_shape[0], self.filter_shape[1], padding=self.padding,
                                stride=self.stride)
        x_cols_argmax = np.argmax(x_cols, axis=0)
        x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
        out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)
        self.cache = (A_pre, x_cols, x_cols_argmax)
        return out

    def backward(self, dZ):
        x, x_cols, x_cols_argmax = self.cache
        del self.cache
        N, C, H, W = x.shape

        dout_reshaped = dZ.transpose(2, 3, 0, 1).flatten()
        dx_cols = np.zeros_like(x_cols)
        dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
        dx = col2im_indices(dx_cols, (N * C, 1, H, W), self.filter_shape[0], self.filter_shape[1],
                            padding=self.padding, stride=self.stride)
        dx = dx.reshape(x.shape)
        return dx

# class Convolution(Layer):
#     def __init__(self, filter_count, filter_shape, stride=1, padding=0, activation='relu', batchNormal=False):
#         super(Convolution, self).__init__(activation=activation)
#         self.name = 'Convolution'
#         self.filter_count = filter_count  # 卷积核数量
#         self.filter_shape = filter_shape  # 卷积核形状
#         self.stride = stride  # 步长
#         self.padding = padding  # pad
#         self.Z_pad = None
#         self.batchNormal = BatchNormal() if batchNormal else None
#
#     def init_params(self, A_pre):  # pre_nc 前一个通道数
#         pre_nc = A_pre.shape[1]
#
#         if self.W is None:
#             W_shape = (self.filter_count, pre_nc, self.filter_shape[0], self.filter_shape[1])
#             n_l = self.filter_shape[0] * self.filter_shape[1] * self.filter_count
#             if self.activation == 'relu':  # 'kaiming'
#                 self.W = np.random.normal(loc=0.0, scale=math.sqrt(2. / n_l), size=W_shape)
#             elif self.activation == 'leak_relu':  # 'kaiming'
#                 self.W = np.random.normal(loc=0.0, scale=math.sqrt(2. / (1.0001 * n_l)), size=W_shape)
#             else:
#                 n_x, d_x, h_x, w_x = A_pre.shape  # 'xavier'
#                 self.W = np.random.normal(loc=0.0, scale=math.sqrt(2. / (pre_nc + d_x)), size=W_shape)
#             self.dW = np.zeros_like(self.W)
#
#         if self.b is None:
#             self.b = np.random.randn(self.filter_count)
#             self.db = np.zeros_like(self.b)
#
#     def forward(self, A_pre, mode='train'):
#         self.init_params(A_pre)
#         self.A_pre = A_pre
#         FN, C, FH, FW = self.W.shape
#         N, C, H, W = A_pre.shape
#         out_h = 1 + int((H + 2 * self.padding - FH) / self.stride)
#         out_w = 1 + int((W + 2 * self.padding - FW) / self.stride)
#
#         col = im2col(A_pre, FH, FW, self.stride, self.padding)
#         col_W = self.W.reshape(FN, -1).T
#
#         out = np.dot(col, col_W) + self.b
#         out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
#
#         self.col = col
#         self.col_W = col_W
#         Z = self.batchNormal.forward(out, mode) if self.batchNormal else out
#         return ac_get(Z, self.activation)
#
#     def backward(self, dout):
#         if self.batchNormal:
#             dout = self.batchNormal.backward(dout)
#         FN, C, FH, FW = self.W.shape
#         dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
#
#         self.db = np.sum(dout, axis=0)
#         self.dW = np.dot(self.col.T, dout)
#         self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)
#
#         dcol = np.dot(dout, self.col_W.T)
#         dx = col2im(dcol, self.A_pre.shape, FH, FW, self.stride, self.padding)
#         return ac_get_grad(dx, self.A_pre, self.activation)
#
#     @property
#     def params(self):
#         return self.W, self.b
#
#     @property
#     def grads(self):
#         return self.dW, self.db
#
#
# class Pooling(Layer):
#     def __init__(self, filter_shape, paddingMode='same', stride=1, mode='max'):
#         super(Pooling, self).__init__()
#         self.filter_shape = filter_shape
#         self.pool_h , self.pool_w = filter_shape
#         self.name = 'Pooling'
#         self.stride = stride
#         self.padding = 0 if paddingMode == 'valid' else (filter_shape[0] - 1) // 2
#         self.mode = mode
#         assert (self.mode in ['max', 'average'])
#
#     def forward(self, x, mode='train'):
#         N, C, H, W = x.shape
#         out_h = int(1 + (H - self.pool_h) / self.stride)
#         out_w = int(1 + (W - self.pool_w) / self.stride)
#
#         col = im2col(x, self.pool_h, self.pool_w, self.stride, self.padding)
#         col = col.reshape(-1, self.pool_h * self.pool_w)
#
#         arg_max = np.argmax(col, axis=1)
#         out = np.max(col, axis=1)
#         out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
#
#         self.x = x
#         self.arg_max = arg_max
#
#         return out
#
#     def backward(self, dout):
#         dout = dout.transpose(0, 2, 3, 1)
#
#         pool_size = self.pool_h * self.pool_w
#         dmax = np.zeros((dout.size, pool_size))
#         dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
#         dmax = dmax.reshape(dout.shape + (pool_size,))
#
#         dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
#         dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.padding)
#
#         return dx


# class ConvolutionForloop(Layer):
#
#     def __init__(self, filter_count, filter_shape, stride=1, paddingMode='same', activation='relu', batchNormal=False):
#         super(ConvolutionForloop, self).__init__(activation=activation)
#         self.filter_count = filter_count  # 卷积核数量
#         self.filter_shape = filter_shape  # 卷积核形状
#         self.stride = stride  # 步长
#         self.padding = 0 if paddingMode == 'valid' else (filter_shape[0] - 1) // 2
#         self.Z_pad = None
#         self.batchNormal = BatchNormal() if batchNormal else None
#
#     def init_params(self, pre_nc):  # pre_nc 前一个通道数
#         if self.W is None:
#             kernel_shape = (self.filter_count, pre_nc, self.filter_shape[0], self.filter_shape[1])
#             self.W = np.random.randn(*kernel_shape)  # W.shape == (f, f ,pre_nc, nc)
#         if self.b is None:
#             self.b = np.random.randn(self.filter_count)  # b.shape = (1, 1, 1, nc)
#
#     # 没问题
#     def forward(self, A_pre, mode='train'):
#         self.A_pre = A_pre
#         self.init_params(A_pre.shape[1])
#
#         n_h = int((A_pre.shape[2] + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
#         n_w = int((A_pre.shape[3] + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
#         Z = np.zeros((A_pre.shape[0], self.filter_count, n_h, n_w))
#
#         self.Z_pad = ZeroPad(A_pre, self.padding) if self.padding > 0 else A_pre
#
#         for i in range(A_pre.shape[0]):
#             a_prev_pad = self.Z_pad[i]
#             for h in range(Z.shape[2]):
#                 for w in range(Z.shape[3]):
#                     for nc in range(self.filter_count):
#                         vs = h * self.stride
#                         ve = vs + self.filter_shape[0]
#                         hs = w * self.stride
#                         he = hs + self.filter_shape[1]
#                         a_slice = a_prev_pad[:, vs:ve, hs:he]
#                         Z[i, nc, h, w] = np.sum(a_slice * self.W[nc, :, :, :] + self.b[nc])
#         Zhat = self.batchNormal.forward(Z, mode) if self.batchNormal else Z
#         return ac_get(Zhat, self.activation)
#
#     # 没问题
#     def backward(self, dZ):
#         if self.batchNormal:
#             dZ = self.batchNormal.backward(dZ)
#         dZ_pad = np.zeros_like(self.Z_pad)
#         m, nc, n_h, n_w = dZ.shape
#         self.dW = np.zeros_like(self.W)
#         self.db = np.zeros_like(self.b)
#         dA = np.zeros_like(self.A_pre)
#
#         if self.padding > 0:
#             for i in range(m):
#                 a_pre = self.Z_pad[i]
#                 da_pre = dZ_pad[i]
#                 for h in range(n_h):
#                     for w in range(n_w):
#                         for c in range(nc):
#                             vs = h * self.stride
#                             ve = vs + self.filter_shape[0]
#                             hs = w * self.stride
#                             he = hs + self.filter_shape[1]
#                             a_slice = a_pre[:, vs:ve, hs:he]
#                             da_pre[:, vs:ve, hs:he] = da_pre[:, vs:ve, hs:he] + self.W[c, :, :, :] * dZ[i, c, h, w]
#                             self.dW[c, :, :, :] = self.dW[c, :, :, :] + a_slice * dZ[i, c, h, w]
#                             self.db[c] = self.db[c] + dZ[i, c, h, w]
#                 dA[i] = da_pre[:, self.padding:-self.padding, self.padding:-self.padding]
#         else:
#             for i in range(m):
#                 a_pre = self.Z_pad[i]
#                 da_pre = dZ_pad[i]
#                 for h in range(n_h):
#                     for w in range(n_w):
#                         for c in range(nc):
#                             vs = h * self.stride
#                             ve = vs + self.filter_shape[0]
#                             hs = w * self.stride
#                             he = hs + self.filter_shape[1]
#                             a_slice = a_pre[:, vs:ve, hs:he]
#                             da_pre[:, vs:ve, hs:he] = da_pre[:, vs:ve, hs:he] + self.W[c, :, :, :] * dZ[i, c, h, w]
#                             self.dW[c, :, :, :] = self.dW[c, :, :, :] + a_slice * dZ[i, c, h, w]
#                             self.db[c] = self.db[c] + dZ[i, c, h, w]
#                 dA[i] = da_pre
#         return ac_get_grad(dA, self.A_pre, self.activation)
#
#     @property
#     def params(self):
#         return self.W, self.b
#
#     @property
#     def grads(self):
#         return self.dW, self.db


#
# class PoolingForloop(Layer):
#     def __init__(self, filter_shape, paddingMode='same', stride=1, mode='max'):
#         super(PoolingForloop, self).__init__()
#         self.filter_shape = filter_shape
#         self.padding = 0 if paddingMode == 'valid' else (filter_shape[0] - 1) // 2
#         self.stride = stride
#         self.mode = mode
#         assert (self.mode in ['max', 'average'])
#
#     def init_params(self, nx):
#         pass
#
#     def forward(self, A_pre, mode='train'):
#         m, nc, h, w = A_pre.shape
#
#         n_h = int((h + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
#         n_w = int((w + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
#         A = np.zeros((m, nc, n_h, n_w))
#
#         self.A_pad = ZeroPad(A_pre, self.padding) if self.padding > 0 else A_pre
#
#         if self.mode == 'max':
#             for i in range(m):
#                 a_prev_pad = self.A_pad[i]
#                 for h in range(n_h):
#                     for w in range(n_w):
#                         for c in range(nc):
#                             vs = h * self.stride
#                             ve = vs + self.filter_shape[0]
#                             hs = w * self.stride
#                             he = hs + self.filter_shape[1]
#                             a_slice = a_prev_pad[c, vs:ve, hs:he]
#                             A[i, c, h, w] = np.max(a_slice)
#         elif self.mode == 'average':
#             for i in range(m):
#                 a_prev_pad = self.A_pad[i]
#                 for h in range(n_h):
#                     for w in range(n_w):
#                         for c in range(nc):
#                             vs = h * self.stride
#                             ve = vs + self.filter_shape[0]
#                             hs = w * self.stride
#                             he = hs + self.filter_shape[1]
#                             a_slice = a_prev_pad[c, vs:ve, hs:he]
#                             A[i, c, h, w] = np.mean(a_slice)
#         return A
#
#     def backward(self, dZ):
#         m, nc, n_h, n_w = dZ.shape
#         dA = np.zeros_like(self.A_pad)
#         if self.mode == 'max':
#             for i in range(m):
#                 da = dZ[i]
#                 for h in range(n_h):
#                     for w in range(n_w):
#                         for c in range(nc):
#                             vs = h * self.stride
#                             ve = vs + self.filter_shape[0]
#                             hs = w * self.stride
#                             he = hs + self.filter_shape[1]
#                             dA[i, c, vs:ve, hs:he] += self.maxPooling_backward(self.A_pad[i, c, vs:ve, hs:he],
#                                                                                da[c, h, w])
#         elif self.mode == 'average':
#             for i in range(m):
#                 da = dZ[i]
#                 for h in range(n_h):
#                     for w in range(n_w):
#                         for c in range(nc):
#                             vs = h * self.stride
#                             ve = vs + self.filter_shape[0]
#                             hs = w * self.stride
#                             he = hs + self.filter_shape[1]
#                             dA[i, c, vs:ve, hs:he] += self.averagePooling_backward(da[c, h, w])
#
#         return dA[:, :, self.padding:-self.padding, self.padding:-self.padding] if self.padding > 0 else dA
#
#     def maxPooling_backward(self, z, grad):  # input: Z:matrix, grad is real return matrix
#         assert (z.ndim == 2)
#         return (z == np.max(z)) * grad
#
#     def averagePooling_backward(self, a):  # input : a is real return matrix
#         return np.ones((self.filter_shape[0], self.filter_shape[1])) * (
#                 a / (self.filter_shape[0] * self.filter_shape[1]))
