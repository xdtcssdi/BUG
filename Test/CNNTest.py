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
def f2():
    # 预处理数据

    # %% md
    seed = 100
    nb_data = 1000
    print("loading data ....")
    mnist = fetch_openml('mnist_784', version=1, cache=True)
    X_train = mnist.data.reshape((-1, 28, 28, 1)) / 255.0
    np.random.seed(seed)
    X_train = np.random.permutation(X_train)[:nb_data]
    y_train = mnist.target
    np.random.seed(seed)
    y_train = np.random.permutation(y_train)[:nb_data].reshape((-1,))
    n_classes = np.unique(y_train).size
    print('mnist.data shape : ', mnist.data.shape)
    print('X_train shape : ', X_train.shape)
    print('Y_train shape : ', y_train.shape)

    net = Model()
    net.add(Convolution(1, (3, 3), pre_nc=1, activation='relu'))
    net.add(Pooling((2, 2), stride=2))
    net.add(Convolution(2, (4, 4), pre_nc=1, activation='relu'))
    net.add(Pooling((2, 2), stride=2))
    net.add(Flatten())
    net.add(Core(10, activation='Softmax').init_params(50))
    net.complie()
    # %% code

    net.train(X_train, one_hot(y_train), printLoss=True)


def one_hot(labels, nb_classes=None):
    classes = np.unique(labels)
    if nb_classes is None:
        nb_classes = classes.size
    one_hot_labels = np.zeros((labels.shape[0], nb_classes))

    for i, c in enumerate(classes):
        one_hot_labels[labels == c, i] = 1
    return one_hot_labels

def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)


if __name__ == '__main__':
    np.random.seed(1)
    f1()