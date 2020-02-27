from tensorflow import keras

from BUG.Layers.Layer import Convolution, Pooling, Core
from BUG.Model.Model import Model
from BUG.function.util import one_hot
from BUG.load_package import p


def LeNet5():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    X_train = p.reshape(train_images, (train_images.shape[0], 1, 28, 28)) / 255.
    X_test = p.reshape(test_images, (test_images.shape[0], 1, 28, 28))[:10000] / 255.
    Y_train = one_hot(train_labels, 10)
    Y_test = one_hot(test_labels, 10)[:10000]
    net = Model()
    net.add(Convolution(filter_count=6, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='valid'))
    net.add(Convolution(filter_count=16, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='valid'))
    net.add(Core(120, batchNormal=True))
    net.add(Core(84, batchNormal=True))
    net.add(Core(10, batchNormal=True, activation="softmax"))
    net.compile()
    net.fit(X_train, Y_train, X_test, Y_test, batch_size=100000,
            learning_rate=0.0075, is_normalizing=False,
            lossMode='SoftmaxCrossEntry', optimize='Adam')


if __name__ == '__main__':
    p.random.seed(1)
    LeNet5()
