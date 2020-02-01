from Activation import *
from Layers.Core import Core
from Loss import *
from Model import Model
from util import *


def f():
    # 预处理数据
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T/255
    Y_train = train_set_y_orig

    m = X_train.shape[1]
    D = (X_train.shape[0], 7, 1)  # 两层网络
    Lc = len(D)

    # 初始化W，b
    params = {}
    #np.random.seed(1)
    for i in range(1, Lc):  # 1,2
        params['W'+str(i)] = np.random.randn(D[i], D[i-1]) * 0.01
        params['b'+str(i)] = np.zeros((D[i], 1))

    iterator = 2500

    # 设置超参数
    learning_rate = 0.0075

    # 缓存
    caches = {"A0": X_train}
    grads = {}

    for i in range(iterator):

        # 前向传播
        A = X_train

        for j in range(1, Lc):  # 1,2
            Z = np.dot(params['W'+str(j)], A) + params['b'+str(j)]
            if j == 1:
                A = Relu(Z)
            else:
                A = Sigmoid(Z)

            caches['Z' + str(j)] = Z
            caches['A' + str(j)] = A

        Y_hat = A

        # 计算损失
        loss = CrossEntry(Y_train, Y_hat)

        # 打印损失

        if i % 100 == 0:
            print(loss)

        # 反向传播

        for j in reversed(range(1, Lc)):  # 2,1
            if j == Lc - 1:
                dZ = CrossEntryGrad(Y_train, Y_hat) * SigmoidGrad(caches['Z' + str(Lc-1)])  # 最后一层dL/dz
            else:
                dZ = np.dot(params['W'+str(j+1)].T, grads['dZ'+str(j+1)]) * ReluGrad(caches['Z'+str(j)])

            dW = 1. / m * np.dot(dZ, caches['A' + str(j - 1)].T)
            db = np.mean(dZ, axis=1, keepdims=True)
            grads['dZ' + str(j)] = dZ
            grads['dW' + str(j)] = dW
            grads['db' + str(j)] = db

        # 更新参数

        for j in reversed(range(1, Lc)):
            params['W' + str(j)] = params['W' + str(j)] - learning_rate * grads['dW' + str(j)]
            params['b' + str(j)] = params['b' + str(j)] - learning_rate * grads['db' + str(j)]
def f1():
    np.random.seed(1)

    # 数据预处理
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T / 255
    Y_train = train_set_y_orig

    # 创建网络架构
    net = Model()
    net.add(Core(7, "relu"))
    net.add(Core(1, "sigmoid"))
    net.complie(X_train)
    net.train(X_train, Y_train, printLoss=True,learning_rate=0.0075,iterator=2500)


if __name__ == '__main__':
    f1()