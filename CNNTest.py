from BUG.Layers.Layer import ConvolutionForloop, PoolingForloop, Flatten, Core
from BUG.Model.Model import Model
from BUG.function.util import one_hot, load_dataset
import numpy as cp


def LeNet5():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
    X_train = train_set_x_orig.transpose(0, 3, 1, 2)
    Y_train = one_hot(train_set_y_orig.reshape(train_set_y_orig.shape[-1]))
    X_test = test_set_x_orig.transpose(0, 3, 1, 2)
    Y_test = one_hot(test_set_y_orig.reshape(test_set_y_orig.shape[-1]))

    net = Model()
    net.add(ConvolutionForloop(filter_count=5, filter_shape=(5, 5), batchNormal=True))
    net.add(PoolingForloop(filter_shape=(2, 2), stride=2, mode='max', paddingMode='valid'))
    # net.add(Convolution(filter_count=16, filter_shape=(5, 5), batchNormal=False))
    # net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='valid'))
    net.add(Flatten())
    # net.add(Core(120))
    net.add(Core(32))
    net.add(Core(len(classes), activation="softmax"))
    net.compile()
    net.train(X_train, Y_train, X_test, Y_test, batch_size=100, learning_rate=0.0075,
              validation_percentage=0, testing_percentage=0,
              lossMode='SoftmaxCrossEntry', optimize='Adam')


if __name__ == '__main__':
    cp.random.seed(1)
    LeNet5()

