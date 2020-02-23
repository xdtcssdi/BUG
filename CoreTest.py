import cupy as cp

from BUG_GPU.Layers.Layer import Core
from BUG_GPU.Model.Model import Model
from BUG_GPU.function.util import load_data


def f2():
    cp.random.seed(1)
    # 数据预处理
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).astype(cp.float64)
    Y_train = train_set_y_orig.T

    X_test = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).astype(cp.float64)
    Y_test = test_set_y_orig.T

    # 创建网络架构
    net = Model()
    #net.add(Core(20, batchNormal=True))
    net.add(Core(7, batchNormal=False))
    #net.add(Core(5, batchNormal=False))
    net.add(Core(1, "sigmoid"))
    net.compile()
    net.train(X_train, Y_train, X_test, Y_test, batch_size=12,
              testing_percentage=0, validation_percentage=0,
              learning_rate=0.0075, iterator=2500, optimize='Adam')


if __name__ == '__main__':
    f2()
