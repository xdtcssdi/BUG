import numpy as np

from BUG.Layers.Layer import Dense
from BUG.Model.model import Linear_model
from BUG.function.util import load_data, one_hot
from tensorflow import keras
import matplotlib.pyplot as plt
from BUG.load_package import p

def f2():
    # 数据预处理
    path = ['/Users/oswin/Documents/BS/BUG/datasets/train_catvnoncat.h5',
            '/Users/oswin/Documents/BS/BUG/datasets/test_catvnoncat.h5']
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data(path)
    X_train = train_set_x_orig
    X_test = test_set_x_orig
    Y_train = train_set_y_orig.T
    Y_test = test_set_y_orig.T

    # 创建网络架构
    net = Linear_model()
    net.add(Dense(64, batchNormal=True, flatten=True))
    net.add(Dense(16, batchNormal=False))
    net.add(Dense(8, batchNormal=False))
    net.add(Dense(1, "sigmoid"))
    net.compile()
    net.fit(X_train, Y_train, X_test, Y_test, batch_size=X_train.shape[0]//10,
            is_normalizing=False, lossMode='CrossEntry',save_epoch=1000,
            learning_rate=0.001, iterator=1000, optimize='BGD')


def f3():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig
    X_test = test_set_x_orig
    Y_train = train_set_y_orig.T
    Y_test = test_set_y_orig.T

    net = LinearModel()
    net.load_model('model.h5')
    res = net.predict(X_test)
    print(res)


if __name__ == '__main__':
    np.random.seed(1)
    f2()
