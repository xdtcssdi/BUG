from Activation import *
from Layers.Core import Core
from Loss import *
from Model import Model
from util import *


def f1():
    np.random.seed(1)
    # 数据预处理
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1) / 255.
    Y_train = train_set_y_orig.T
    X_test = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1) / 255.
    Y_test = test_set_y_orig.T
    # 创建网络架构
    net = Model()
    net.add(Core(20, batchNormal=True))
    net.add(Core(7))
    #net.add(Core(5))
    net.add(Core(1, "sigmoid"))
    net.compile()
    net.train(X_train, Y_train, X_test, Y_test, normalizing_inputs=False, batch_size=100,
              testing_percentage=0, validation_percentage=0,
              printLoss=True, learning_rate=0.0075, iterator=2500, printOneTime=True)


def f2():
    np.random.seed(1)
    # 数据预处理
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1) / 255.
    Y_train = train_set_y_orig.T

    X_test = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1) / 255.
    Y_test = test_set_y_orig.T

    # 创建网络架构
    net = Model()
    net.add(Core(31, batchNormal=True))
    net.add(Core(20, batchNormal=True))
    net.add(Core(7, batchNormal=False))
    net.add(Core(5, batchNormal=False))
    net.add(Core(1, "sigmoid"))
    net.compile()
    net.train(X_train, Y_train, X_test, Y_test, normalizing_inputs=False, batch_size=64,
              testing_percentage=0, validation_percentage=0,
              printLoss=True, learning_rate=0.0075, iterator=2500)


if __name__ == '__main__':
    f2()

