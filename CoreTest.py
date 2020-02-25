import numpy as np

from BUG.Layers.Layer import Core
from BUG.Model.Model import Model
from BUG.function.util import load_data, one_hot
from tensorflow import keras

def f2():
    # 数据预处理

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig
    X_test = test_set_x_orig
    Y_train = train_set_y_orig.T
    Y_test = test_set_y_orig.T

    # 创建网络架构
    net = Model()
    net.add(Core(64, batchNormal=True))
    net.add(Core(32, batchNormal=False))
    net.add(Core(16, batchNormal=False))
    net.add(Core(1, "sigmoid"))
    net.compile()
    net.fit(X_train, Y_train, X_test, Y_test, batch_size=X_train.shape[0]//10,is_normalizing=False,
              testing_percentage=0, validation_percentage=0, lossMode='CrossEntry',
              learning_rate=0.001, iterator=1000, optimize='Adam')


def f3():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig
    X_test = test_set_x_orig
    Y_train = train_set_y_orig.T
    Y_test = test_set_y_orig.T

    net = Model()
    net.load_model('model.h5')
    res = net.predict(X_test)
    print(res)


if __name__ == '__main__':
    np.random.seed(1)
    f2()
