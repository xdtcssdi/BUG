import sys

from .Normalization import BatchNormal

sys.path.append("../")
import Activation
from .Layer import Layer
from .Padding import *


class ConvolutionForloop(Layer):

    def __init__(self, filter_count, filter_shape, stride=1, padding=0, activation='relu', batchNormal=False):
        super(ConvolutionForloop, self).__init__(activation=activation)
        self.filter_count = filter_count  # 卷积核数量
        self.filter_shape = filter_shape  # 卷积核形状
        self.stride = stride  # 步长
        self.padding = padding  # pad
        self.Z_pad = None
        self.batchNormal = BatchNormal() if batchNormal else None

    def init_params(self, pre_nc):  # pre_nc 前一个通道数
        if self.W is None:
            kernel_shape = (self.filter_shape[0], self.filter_shape[1], pre_nc, self.filter_count)
            self.W = np.random.randn(*kernel_shape)  # W.shape == (f, f ,pre_nc, nc)
        if self.b is None:
            b_shape = [1, ] * self.W.ndim
            b_shape[-1] = self.filter_count
            self.b = np.random.randn(*tuple(b_shape))  # b.shape = (1, 1, 1, nc)

    # 没问题
    def forward(self, A_pre, mode='train'):
        self.A_pre = A_pre
        self.init_params(A_pre.shape[-1])
        n_w = int((A_pre.shape[1] + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
        n_h = int((A_pre.shape[2] + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
        Z = np.zeros((A_pre.shape[0], n_w, n_h, self.filter_count))

        self.Z_pad = ZeroPad(A_pre, self.padding) if self.padding > 0 else A_pre

        for i in range(A_pre.shape[0]):
            a_prev_pad = self.Z_pad[i]
            for w in range(Z.shape[1]):
                for h in range(Z.shape[2]):
                    for nc in range(self.filter_count):
                        hs = w * self.stride
                        he = hs + self.filter_shape[0]
                        vs = h * self.stride
                        ve = vs + self.filter_shape[1]
                        a_slice = a_prev_pad[hs:he, vs:ve, :]
                        Z[i, w, h, nc] = np.sum(a_slice * self.W[:, :, :, nc] + self.b[:, :, :, nc])
        Zhat = self.batchNormal.forward(Z, mode) if self.batchNormal else Z
        return Activation.get(Zhat, self.activation)

    # 没问题
    def backward(self, dZ):
        if self.batchNormal:
            dZ = self.batchNormal.backward(dZ)
        dZ_pad = np.zeros_like(self.Z_pad)
        m, n_w, n_h, nc = dZ.shape
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        dA = np.zeros_like(self.A_pre)

        if self.padding > 0:
            for i in range(m):
                a_pre = self.Z_pad[i]
                da_pre = dZ_pad[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            a_slice = a_pre[hs:he, vs:ve, :]
                            da_pre[hs:he, vs:ve, :] += self.W[:, :, :, c] * dZ[i, w, h, c]
                            self.dW[:, :, :, c] += a_slice * dZ[i, w, h, c]
                            self.db[:, :, :, c] += dZ[i, w, h, c]
                dA[i] = da_pre[self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            for i in range(m):
                a_pre = self.Z_pad[i]
                da_pre = dZ_pad[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            a_slice = a_pre[hs:he, vs:ve, :]
                            da_pre[hs:he, vs:ve, :] += self.W[:, :, :, c] * dZ[i, w, h, c]
                            self.dW[:, :, :, c] += a_slice * dZ[i, w, h, c]
                            self.db[:, :, :, c] += dZ[i, w, h, c]
                dA[i] = da_pre
        return Activation.get_grad(dA, self.A_pre, self.activation)

    @property
    def params(self):
        return self.W, self.b

    @property
    def grads(self):
        return self.dW, self.db


class Convolution(Layer):

    def __init__(self, filter_count, filter_shape, stride=1, padding=0, activation='relu', batchNormal=False):
        super(Convolution, self).__init__(activation=activation)
        self.name = 'Convolution'
        self.filter_count = filter_count  # 卷积核数量
        self.filter_shape = filter_shape  # 卷积核形状
        self.stride = stride  # 步长
        self.padding = padding  # pad
        self.Z_pad = None
        self.batchNormal = BatchNormal() if batchNormal else None

    def init_params(self, pre_nc):  # pre_nc 前一个通道数
        if self.W is None:
            self.W = np.random.randn(self.filter_count, pre_nc,
                                     self.filter_shape[0], self.filter_shape[1])  # W.shape == (nc, pre_nc, f, f)
            self.dW = np.zeros_like(self.W)

        if self.b is None:
            self.b = np.random.randn(self.filter_count)
            assert self.b.shape == (self.filter_count,)
            self.db = np.zeros_like(self.b)

    # 没问题
    def forward(self, A_pre, mode='train'):
        self.init_params(A_pre.shape[1])
        self.A_pre = A_pre
        output_data = self.conv(A_pre, self.W, self.b, self.stride, self.padding)
        Z = self.batchNormal.forward(output_data, mode) if self.batchNormal else output_data
        return Activation.get(Z, self.activation)

    # 没问题
    def backward(self, dZ):
        if self.batchNormal:
            dZ = self.batchNormal.backward(dZ)
        self.db = np.sum(dZ, axis=(0, 2, 3))
        num_filters, _, filter_height, filter_width = self.W.shape
        dout_reshaped = dZ.transpose(1, 2, 3, 0).reshape(num_filters, -1)
        self.dW = dout_reshaped.dot(self.X_col.T).reshape(self.W.shape)
        dx_cols = self.W.reshape(num_filters, -1).T.dot(dout_reshaped)
        dx = self.col2im_indices(dx_cols, self.A_pre.shape, filter_height, filter_width, self.padding, self.stride)
        return Activation.get_grad(dx, self.A_pre, self.activation)

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
        self.X_col = self.im2col_indices(X, kernel_size, kernel_size, padding=padding, stride=stride)
        W_col = W.reshape(n_filters, -1)
        out = (np.dot(W_col, self.X_col).T + b).T
        out = out.reshape(n_filters, h_out, w_out, n_x)
        out = out.transpose(3, 0, 1, 2)
        return out

    def get_im2col_indices(self, x_shape, field_height, field_width, padding=1, stride=1):
        # First figure out what the size of the output should be
        N, C, H, W = x_shape
        assert (H + 2 * padding - field_height) % stride == 0
        assert (W + 2 * padding - field_width) % stride == 0
        out_height = (H + 2 * padding - field_height) // stride + 1
        out_width = (W + 2 * padding - field_width) // stride + 1
        i0 = np.repeat(np.arange(field_height), field_width)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_height), out_width)
        j0 = np.tile(np.arange(field_width), field_height * C)
        j1 = stride * np.tile(np.arange(out_width), out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

        return k, i, j

    def im2col_indices(self, x, field_height, field_width, padding=1, stride=1):
        """ An implementation of im2col based on some fancy indexing """
        # Zero-pad the input
        p = padding
        x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        k, i, j = self.get_im2col_indices(x.shape, field_height, field_width, padding, stride)

        cols = x_padded[:, k, i, j]
        C = x.shape[1]
        cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
        return cols

    def col2im_indices(self, cols, x_shape, field_height=3, field_width=3, padding=1,
                       stride=1):
        """ An implementation of col2im based on fancy indexing and np.add.at """
        N, C, H, W = x_shape
        H_padded, W_padded = H + 2 * padding, W + 2 * padding
        x_padded = np.zeros((N, C, H_padded, H_padded), dtype=cols.dtype)
        k, i, j = self.get_im2col_indices(x_shape, field_height, field_width, padding, stride)
        cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
        if padding == 0:
            return x_padded
        return x_padded[:, :, padding:-padding, padding:-padding]
