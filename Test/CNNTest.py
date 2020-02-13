from Layers.Convolution import Convolution
from Layers.Core import Core
from Layers.Flatten import Flatten
from Layers.Pooling import Pooling
from Model import Model
from util import *


def f3():
    # mnist = fetch_openml("mnist_784", cache=True, version=1)
    # print(mnist)
    # X_train = mnist.data.reshape(mnist["data"].shape[0], 28, 28, 1)[:1000] / 255.
    # Y_train = mnist.target.astype(np.float64)[:1000]

    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig.transpose(0, 3, 1, 2)
    Y_train = train_set_y_orig.reshape((-1, 1)).astype(np.float64)
    X_test = test_set_x_orig.transpose(0, 3, 1, 2)
    Y_test = test_set_y_orig.reshape((-1, 1)).astype(np.float64)

    net = Model()
    net.add(Convolution(filter_count=3, filter_shape=(7, 7), activation='relu', batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max'))
    net.add(Convolution(filter_count=1, filter_shape=(4, 4), activation='relu'))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max'))
    net.add(Flatten())
    net.add(Core(33))
    net.add(Core(1, activation="sigmoid"))
    net.compile()
    log_file = open("log.txt", 'w+')
    net.train(X_train, Y_train, X_test, Y_test, batch_size=64, validation_percentage=0, testing_percentage=0,
              printLoss=True, lossMode='SoftmaxCrossEntry', tms=10, log=log_file, )


def LeNet5():
    num = 10000
    #  train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train, Y_train, X_test, Y_test = load_CIFAR10(r"/Users/oswin/Documents/BUG/datasets/cifar-10-python")
    X_train = X_train.transpose(0, 3, 1, 2)[:num]
    print(X_train.shape)
    Y_train = Y_train.reshape((-1, 1)).astype(np.float64)[:num]
    X_test = X_test.transpose(0, 3, 1, 2)[:num]
    Y_test = Y_test.reshape((-1, 1)).astype(np.float64)[:num]

    net = Model()
    net.add(Convolution(filter_count=6, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='average'))
    net.add(Convolution(filter_count=16, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='average'))
    net.add(Flatten())
    net.add(Core(120))
    net.add(Core(84))
    net.add(Core(10, activation="softmax"))
    net.compile()
    log_file = open("log.txt", 'w+')
    net.train(X_train, Y_train, X_test, Y_test,
              batch_size=64, learning_rate=0.0075, printOneTime=True,
              validation_percentage=0, testing_percentage=0,
              printLoss=True, lossMode='SoftmaxCrossEntry',
              tms=1, log=log_file)


if __name__ == '__main__':
    np.random.seed(1)
    f3()
