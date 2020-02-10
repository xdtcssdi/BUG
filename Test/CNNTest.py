from Layers.Convolution import Convolution
from Layers.Core import Core
from Model import Model
from util import *
from Layers.Pooling import Pooling
from Layers.Flatten import Flatten


def f3():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_data()
    X_train = train_set_x_orig / 255.
    Y_train = train_set_y_orig.reshape((-1, 1)).astype(np.float64)

    net = Model()
    net.add(Convolution(filter_count=3, filter_shape=(3, 3), activation='relu'))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='average'))
    net.add(Convolution(filter_count=1, filter_shape=(4, 4), activation='relu'))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='average'))
    net.add(Flatten())
    net.add(Core(30, activation="relu"))
    net.add(Core(1, activation="sigmoid"))
    net.compile()
    log_file = open("log.txt",'w+')
    net.train(X_train, Y_train, printLoss=True, lossMode='CrossEntry', tms=1, log=log_file)


if __name__ == '__main__':
    np.random.seed(100)
    f3()

