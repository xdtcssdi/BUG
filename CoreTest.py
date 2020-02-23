import numpy as cp

from BUG.Layers.Layer import Core, Flatten
from BUG.Model.Model import Model
from BUG.function.util import load_data, one_hot
from tensorflow import keras

def f2():
    cp.random.seed(1)
    # 数据预处理

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images.reshape(train_images.shape[0], -1) / 255.
    test_images = test_images.reshape(test_images.shape[0], -1) / 255.
    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)

    # 创建网络架构
    net = Model()

    net.add(Core(128, batchNormal=True))
    net.add(Core(64, batchNormal=True))
    #net.add(Core(5, batchNormal=False))
    net.add(Core(10, "softmax"))
    net.compile()
    net.train(train_images, train_labels, test_images, test_labels, batch_size=6000,
              testing_percentage=0, validation_percentage=0, lossMode='SoftmaxCrossEntry',normalizing_inputs=True,
              learning_rate=0.0075, iterator=2500, optimize='Adam')


if __name__ == '__main__':
    f2()
