import numpy as np

#
# def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
#     # First figure out what the size of the output should be
#     N, C, H, W = x_shape
#     assert (H + 2 * padding - field_height) % stride == 0
#     assert (W + 2 * padding - field_width) % stride == 0
#     out_height = (H + 2 * padding - field_height) // stride + 1
#     out_width = (W + 2 * padding - field_width) // stride + 1
#     i0 = np.repeat(np.arange(field_height), field_width)
#     i0 = np.tile(i0, C)
#     i1 = stride * np.repeat(np.arange(out_height), out_width)
#     j0 = np.tile(np.arange(field_width), field_height * C)
#     j1 = stride * np.tile(np.arange(out_width), out_height)
#     i = i0.reshape(-1, 1) + i1.reshape(1, -1)
#     j = j0.reshape(-1, 1) + j1.reshape(1, -1)
#
#     k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
#
#     return k, i, j
#
#
# def im2col_indices(x, field_height, field_width, padding=1, stride=1):
#     """ An implementation of im2col based on some fancy indexing """
#     # Zero-pad the input
#     p = padding
#     x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
#
#     k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
#
#     cols = x_padded[:, k, i, j]
#     C = x.shape[1]
#     cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
#     return cols
#
#
# def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
#                    stride=1):
#     """ An implementation of col2im based on fancy indexing and np.add.at """
#     N, C, H, W = x_shape
#     H_padded, W_padded = H + 2 * padding, W + 2 * padding
#     x_padded = np.zeros((N, C, H_padded, H_padded), dtype=cols.dtype)
#     k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
#     cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
#     cols_reshaped = cols_reshaped.transpose(2, 0, 1)
#     np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
#     if padding == 0:
#         return x_padded
#     return x_padded[:, :, padding:-padding, padding:-padding]

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充
    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]