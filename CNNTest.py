from BUG_GPU.Layers.Layer import Convolution, Pooling, Core
from BUG_GPU.Model.Model import Model
from BUG_GPU.function.util import one_hot, load_dataset
import cupy as cp
from tensorflow import keras

def LeNet5():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    #print(train_images.shape,train_labels.shape)
    X_train = cp.reshape(train_images, (train_images.shape[0],1,28,28)) / 255.
    X_test = cp.reshape(test_images, (test_images.shape[0], 1,28,28))[:10000] / 255.
    Y_train=one_hot(train_labels, 10)
    Y_test = one_hot(test_labels, 10)[:10000]
    net = Model()
    net.add(Convolution(filter_count=6, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='valid'))
    net.add(Convolution(filter_count=16, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='valid'))
    net.add(Core(120, batchNormal=True))
    net.add(Core(84, batchNormal=True))
    net.add(Core(32, batchNormal=True))
    net.add(Core(10, batchNormal=True, activation="softmax"))
    net.compile()
    net.train(X_train, Y_train, X_test, Y_test, batch_size=10000,
              learning_rate=0.0001,normalizing_inputs=False,
              validation_percentage=0, testing_percentage=0,
              lossMode='SoftmaxCrossEntry', optimize='Adam')


if __name__ == '__main__':
    cp.random.seed(1)
    LeNet5()

