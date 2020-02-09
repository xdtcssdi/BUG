import Activation
from Activation import *

from Layers.Convelution import Convolution
from Layers.Core import Core
from Loss import *
from Model import Model
from util import *
from Layers.Pooling import Pooling
from Layers.Flatten import Flatten
import warnings


from sklearn.datasets import fetch_openml


def one_hot(labels, nb_classes=None):
    classes = np.unique(labels)
    if nb_classes is None:
        nb_classes = classes.size
    one_hot_labels = np.zeros((labels.shape[0], nb_classes))

    for i, c in enumerate(classes):
        one_hot_labels[labels == c, i] = 1
    return one_hot_labels.reshape(10, -1)


def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)


def f2():
    # 预处理数据

    nb_data = 1000
    print("loading data ....")
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    print(mnist.data.shape)
    X_train = mnist.data.T.reshape((28, 28, 1, -1)) / 255.0
    X_train = np.random.permutation(X_train)[:, :, :, :nb_data]

    y_train = mnist.target
    y_train = np.random.permutation(y_train)[:nb_data].reshape((-1,))

    n_classes = np.unique(y_train).size
    print('mnist.data shape : ', mnist.data.shape)
    print('X_train shape : ', X_train.shape)
    print('Y_train shape : ', y_train.shape)
    net = Model()
    net.add(Convolution(1, (3, 3), activation='relu'))
    net.add(Pooling((2, 2), stride=2))
    net.add(Convolution(2, (4, 4), activation='relu'))
    net.add(Pooling((2, 2), stride=2))
    net.add(Flatten())
    net.add(Core(n_classes, "Softmax"))
    net.compile()

    net.train(X_train, one_hot(y_train), printLoss=True, lossMode='SoftmaxCrossEntry', tms=1)


if __name__ == '__main__':
    np.random.seed(1)
    f2()