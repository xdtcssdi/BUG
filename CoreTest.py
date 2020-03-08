import numpy as np

from BUG.Layers.Layer import Dense
from BUG.Model.model import Linear_model
from BUG.function.evaluate import evaluate_many
from BUG.function.util import load_data, load_mnist


def mnist():
    # 数据预处理
    X_train, y_train, X_test, y_test, classes = load_mnist()

    X_train = X_train.reshape(X_train.shape[0], -1) / 255.
    Y_train = y_train

    X_test = X_test.reshape(X_test.shape[0], -1) / 255.
    y_test = y_test

    accuracy = evaluate_many
    # 创建网络架构
    net = Linear_model()
    net.add(Dense(64, batchNormal=True))
    net.add(Dense(32, batchNormal=True))
    net.add(Dense(16, batchNormal=True))
    net.add(Dense(classes, "softmax"))
    net.compile()
    net.fit(X_train, Y_train, accuracy, X_test, y_test, batch_size=512,
            is_normalizing=False, lossMode='SoftmaxCrossEntry', save_epoch=1000,
            learning_rate=0.005, iterator=1000, optimize='Adam', lambd=0)


if __name__ == '__main__':
    np.random.seed(1)
    mnist()
