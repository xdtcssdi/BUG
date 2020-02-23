import cupy as cp
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1
    i0 = cp.repeat(cp.arange(field_height), field_width)
    i0 = cp.tile(i0, C)
    i1 = stride * cp.repeat(cp.arange(out_height), out_width)
    j0 = cp.tile(cp.arange(field_width), field_height * C)
    j1 = stride * cp.tile(cp.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = cp.repeat(cp.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the icput
    p = padding
    x_padded = cp.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and cp.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = cp.zeros((N, C, H_padded, H_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    k = cp.asnumpy(k)
    i = cp.asnumpy(i)
    j = cp.asnumpy(j)
    x_padded_numpy = cp.asnumpy(x_padded)
    cols_reshaped = cp.asnumpy(cols_reshaped)
    np.add.at(x_padded_numpy, (slice(None), k, i, j), cols_reshaped)
    del k,i,j
    x_padded = cp.asarray(x_padded_numpy)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
