import numpy as np

from BUG.function.util import data_iter_consecutive, one_hot

#
# def one_hot(array, classes):
#     batch_size, time_steps = array.shape
#     one_hots = np.zeros([batch_size, classes, time_steps])
#     for b in range(batch_size):
#         one_hots[b, ] =

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    from math import *

    plt.ion() #开启interactive mode 成功的关键函数
    plt.figure(1)
    t = [0]
    t_now = 0
    m = [sin(t_now)]

    for i in range(2000):
        plt.clf() #清空画布上的所有内容
        t_now = i*0.1
        print(t_now)
        t.append(t_now)#模拟数据增量流入，保存历史数据
        m.append(sin(t_now))#模拟数据增量流入，保存历史数据
        plt.plot(t,m,'-r')
        plt.draw()#注意此函数需要调用
        time.sleep(0.01)
        plt.show()


