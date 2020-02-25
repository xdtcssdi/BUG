import numpy as np

from BUG.Layers.Layer import Core
from BUG.Model.Model import Model
from BUG.function.util import load_data, one_hot
from tensorflow import keras

def f2():
    # 数据预处理

    #fashion_mnist = keras.datasets.fashion_mnist
    #
    # (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # #print(train_images.shape,train_labels.shape)
    # X_train = train_images.reshape(train_images.shape[0], -1) / 255.
    # X_test = test_images.reshape(test_images.shape[0], -1) / 255.
    # Y_train = train_labels.reshape(train_labels.shape[0],1)
    # Y_test = test_labels.reshape(test_labels.shape[0],1)

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig
    X_test = test_set_x_orig
    #X_train = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).astype(np.float32)
    #X_test = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).astype(np.float32)
    Y_train = train_set_y_orig.T
    Y_test = test_set_y_orig.T

    # 创建网络架构
    net = Model()
    net.add(Core(128, batchNormal=True))
    net.add(Core(64, batchNormal=False))
    net.add(Core(32, batchNormal=False))
    net.add(Core(16, batchNormal=False))
    net.add(Core(1, "sigmoid"))
    net.compile()
    net.train(X_train, Y_train, X_test, Y_test, batch_size=X_train.shape[0], normalizing_inputs=False,
              testing_percentage=0, validation_percentage=0, lossMode='CrossEntry',
              learning_rate=0.001, iterator=2500, optimize='Adam')


if __name__ == '__main__':
    np.random.seed(1)
    f2()
