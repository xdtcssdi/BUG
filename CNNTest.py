import matplotlib.pyplot as plt
from tensorflow import keras

from BUG.Layers.Layer import Convolution, Pooling, Dense
from BUG.Model.model import Sequentual
from BUG.function import Loss
from BUG.function.evaluate import evaluate_many
from BUG.load_package import p

label = ['T恤', '裤子', '套衫', '裙子', '外套', '凉鞋', '汗衫', '运动鞋', '包', '踝靴']


def LeNet5():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    X_train = p.array(p.reshape(train_images, (train_images.shape[0], 1, 28, 28)))[:100] / 255.
    X_test = p.array(p.reshape(test_images, (test_images.shape[0], 1, 28, 28)))[:1000] / 255.
    Y_train = p.array(train_labels)[:100]
    Y_test = p.array(test_labels)[:1000]
    net = Sequentual()
    net.add(Convolution(filter_count=32, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='same'))
    # net.add(Convolution(filter_count=64, filter_shape=(5, 5), batchNormal=True))
    # net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='same'))
    net.add(Dense(10, batchNormal=True, flatten=True, activation='relu', keep_prob=0.3))
    net.add(Dense(10, batchNormal=False, activation="softmax"))
    net.compile(lossMode=Loss.SoftCategoricalCross_entropy(), optimize='Adam', accuracy=evaluate_many)
    net.fit(X_train, Y_train, X_test, Y_test, batch_size=10, iterator=1000, learning_rate=0.0075, lambd=0.1,
            path='fashion_mnist_parameters')


def predict():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    idx = 421
    X_test = p.array(p.reshape(test_images, (test_images.shape[0], 1, 28, 28)))[idx] / 255.
    Y_test = test_labels[idx]
    net = Sequentual()
    net.load_model(path='/fashion_mnist_parameters', filename='train_params')
    y_hat = net.predict(X_test.reshape(1, 1, 28, 28))
    print('标签为%s, 识别为%s' % (label[Y_test], label[int(y_hat.argmax(-1))]))

    plt.figure()
    img = X_test.reshape(28, 28) * 255.
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    p.random.seed(1)
    LeNet5()
    # predict()
