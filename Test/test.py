import numpy as np

from BUG.function.util import data_iter_consecutive, one_hot

#
# def one_hot(array, classes):
#     batch_size, time_steps = array.shape
#     one_hots = np.zeros([batch_size, classes, time_steps])
#     for b in range(batch_size):
#         one_hots[b, ] =

if __name__ == '__main__':
    x = np.arange(0, 10).reshape(2,5)
    y = np.arange(10, 20).reshape(2,5)
    print(x, y)
    for x, y in zip(x, y):
        print(x, y)


