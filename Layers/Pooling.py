import numpy as np

from .Layer import Layer
from .Padding import ZeroPad


class PoolingForloop(Layer):
    def __init__(self, filter_shape, padding=0, stride=1, mode='max'):
        super(PoolingForloop, self).__init__()
        self.filter_shape = filter_shape
        self.padding = padding
        self.stride = stride
        self.mode = mode
        assert (self.mode in ['max', 'average'])

    def init_params(self, nx):
        pass

    def forward(self, A_pre, mode='train'):
        m, w, h, nc = A_pre.shape
        n_w = int((w + 2 * self.padding - self.filter_shape[0]) / self.stride + 1)
        n_h = int((h + 2 * self.padding - self.filter_shape[1]) / self.stride + 1)
        A = np.zeros((m, n_w, n_h, nc))

        self.A_pad = ZeroPad(A_pre, self.padding) if self.padding > 0 else A_pre

        if self.mode == 'max':
            for i in range(m):
                a_prev_pad = self.A_pad[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            a_slice = a_prev_pad[hs:he, vs:ve, c]
                            A[i, w, h, c] = np.max(a_slice)
        elif self.mode == 'average':
            for i in range(m):
                a_prev_pad = self.A_pad[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            a_slice = a_prev_pad[hs:he, vs:ve, c]
                            A[i, w, h, c] = np.mean(a_slice)
        return A

    def backward(self, dZ):
        m, n_w, n_h, nc = dZ.shape
        dA = np.zeros_like(self.A_pad)
        if self.mode == 'max':
            for i in range(m):
                da = dZ[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            dA[i, hs:he, vs:ve, c] += self.maxPooling_backward(self.A_pad[i, hs:he, vs:ve, c],
                                                                               da[w, h, c])
        elif self.mode == 'average':
            for i in range(m):
                da = dZ[i]
                for w in range(n_w):
                    for h in range(n_h):
                        for c in range(nc):
                            hs = w * self.stride
                            he = hs + self.filter_shape[0]
                            vs = h * self.stride
                            ve = vs + self.filter_shape[1]
                            dA[i, hs:he, vs:ve, c] += self.averagePooling_backward(da[w, h, c])

        return dA[:, self.padding:-self.padding, self.padding:-self.padding, :] if self.padding > 0 else dA

    def maxPooling_backward(self, z, grad):  # input: Z:matrix, grad is real return matrix
        assert (z.ndim == 2)
        return (z == np.max(z)) * grad

    def averagePooling_backward(self, a):  # input : a is real return matrix
        return np.ones((self.filter_shape[0], self.filter_shape[1])) * (
                a / (self.filter_shape[0] * self.filter_shape[1]))


class Pooling(Layer):
    def __init__(self, filter_shape, padding=0, stride=1, mode='max'):
        super(Pooling, self).__init__()
        self.filter_shape = filter_shape
        self.name = 'Pooling'
        self.padding = padding
        self.stride = stride
        self.mode = mode
        assert (self.mode in ['max', 'average'])

    def init_params(self, nx):
        pass

    def forward(self, A_pre, mode='train'):
        self.input_data = A_pre
        self.num = A_pre.shape[0]
        input_col = self.im2col(self.input_data, self.filter_shape[0], self.stride)
        tmp_index = np.tile(np.arange(input_col.shape[1]), input_col.shape[0]).reshape(input_col.shape)
        self.max_index = tmp_index == input_col.argmax(1).reshape(-1, 1)
        self.output_data = input_col[self.max_index].reshape(self.num, self.input_data.shape[1], self.out_height,
                                                             self.out_width)
        return self.output_data

    def backward(self, dZ):
        diff_col = np.zeros(
            (self.num * self.input_data.shape[1] * self.out_height * self.out_width, self.filter_shape[0] ** 2))
        diff_col[self.max_index] = dZ.reshape(-1)
        diff = self.col2ims(diff_col, self.input_data.shape, self.filter_shape[0], self.stride)
        return diff

    def im2col(self, X, kernel_size=1, stride=1):
        num, channels, height, width = X.shape
        surplus_height = (height - kernel_size) % stride
        surplus_width = (width - kernel_size) % stride
        pad_h = (kernel_size - surplus_height) % kernel_size
        pad_w = (kernel_size - surplus_width) % kernel_size
        X = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        self.X_padded = (num, channels, surplus_height, surplus_width)
        k, i, j = self.get_im2col_indices(X.shape, kernel_size, stride=stride)
        X_col = X[:, k, i, j].reshape(num * channels, kernel_size ** 2, -1)
        X_col = X_col.transpose(0, 2, 1)
        return X_col.reshape(-1, kernel_size ** 2)

    def get_im2col_indices(self, x_shape, kernel_size, padding=0, stride=1):
        N, C, H, W = x_shape
        assert (H + 2 * padding - kernel_size) % stride == 0
        assert (W + 2 * padding - kernel_size) % stride == 0
        self.out_height = (H + 2 * padding - kernel_size) // stride + 1
        self.out_width = (W + 2 * padding - kernel_size) // stride + 1
        i0 = np.repeat(np.arange(kernel_size), kernel_size)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(self.out_height), self.out_width)
        j0 = np.tile(np.arange(kernel_size), kernel_size * C)
        j1 = stride * np.tile(np.arange(self.out_width), self.out_height)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(C), kernel_size * kernel_size).reshape(-1, 1)
        return k.astype(int), i.astype(int), j.astype(int)

    def col2ims(self, x, img_shape, kernel_size, stride):
        x_row_num, x_col_num = x.shape
        img_n, img_c, img_h, img_w = img_shape
        o_h = int(np.math.ceil((img_h - kernel_size + 0.) / stride)) + 1
        o_w = int(np.math.ceil((img_w - kernel_size + 0.) / stride)) + 1
        assert img_n * img_c * o_h * o_w == x_row_num
        assert kernel_size ** 2 == x_col_num
        surplus_h = (img_h - kernel_size) % stride
        surplus_w = (img_w - kernel_size) % stride
        pad_h = (kernel_size - surplus_h) % stride
        pad_w = (kernel_size - surplus_w) % stride
        output_padded = np.zeros((img_n, img_c, img_h + pad_h, img_w + pad_w))
        x_reshape = x.reshape(img_n, img_c, o_h, o_w, kernel_size, kernel_size)
        for n in range(img_n):
            for i in range(o_h):
                for j in range(o_w):
                    output_padded[n, :, i * stride: i * stride + kernel_size, j * stride: j * stride + kernel_size] = \
                        output_padded[n, :, i * stride: i * stride + kernel_size,
                        j * stride: j * stride + kernel_size] + \
                        x_reshape[n, :, i, j, ...]
        return output_padded[:, :, 0: img_h + pad_h, 0: img_w + pad_w]
