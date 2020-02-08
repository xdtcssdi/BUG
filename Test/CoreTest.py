from Activation import *
from Layers.Core import Core
from Loss import *
from Model import Model
from util import *


def f1():
    np.random.seed(1)

    # 数据预处理
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255.
    Y_train = train_set_y_orig

    # 创建网络架构
    net = Model()
    net.add(Core(20))
    net.add(Core(7))
    net.add(Core(5))
    net.add(Core(1, "sigmoid"))
    net.compile()

    net.train(X_train, Y_train, printLoss=True, learning_rate=0.0075, iterator=2500)
    #

    # print()
    # for layer in net.layers:
    #     print(layer.isLast, end=' ')
    # print()
    # for layer in net.layers:
    #     print(layer.pre_layer, end=' ')
    # print()
    # for layer in net.layers:
    #     print(layer.next_layer, end=' ')


if __name__ == '__main__':
    f1()

