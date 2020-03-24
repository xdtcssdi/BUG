import random
import matplotlib.pyplot as plt
from BUG.Layers.Layer import Convolution, Pooling, Dense
from BUG.Model.model import Sequentual
from BUG.function import Loss
from BUG.function.evaluate import evaluate_many
from BUG.function.util import load_mnist
from BUG.load_package import p
from matplotlib.font_manager import FontProperties
font = FontProperties(fname='/Users/oswin/Documents/BS/BUG/datasets/PingFang.ttc', size=16)

label = ['T恤', '裤子', '套衫', '裙子', '外套', '凉鞋', '汗衫', '运动鞋', '包', '踝靴']


def LeNet5():
    train_images, train_labels, test_images, test_labels, classes = load_mnist('/Users/oswin/Documents/BS/'
                                                                          'BUG/datasets/fashion_mnist')
    X_train = p.array(p.reshape(train_images, (train_images.shape[0], 1, 28, 28)))[:1000] / 255.
    X_test = p.array(p.reshape(test_images, (test_images.shape[0], 1, 28, 28)))[:1000] / 255.
    Y_train = p.array(train_labels)[:1000]
    Y_test = p.array(test_labels)[:1000]
    net = Sequentual()
    net.add(Convolution(filter_count=32, filter_shape=(3, 3), batchNormal=True))
    net.add(Convolution(filter_count=64, filter_shape=(3, 3), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='same'))
    net.add(Dense(128, batchNormal=True, flatten=True, activation='relu', keep_prob=0.5))
    net.add(Dense(classes, batchNormal=True, activation="softmax"))
    net.compile(lossMode=Loss.SoftCategoricalCross_entropy(), optimize='Adam', accuracy=evaluate_many)
    net.fit(X_train, Y_train, X_test, Y_test, batch_size=128, iterator=1000, learning_rate=0.0075, is_print=False,
            save_epoch=100000, path='fashion_mnist_parameters1')

def predict():
    _, __, test_images, test_labels, classes = load_mnist('/Users/oswin/Documents/BS/'
                                                     'BUG/datasets/fashion_mnist')
    idx = random.randint(0, test_images.shape[0])
    X_test = p.array(p.reshape(test_images, (test_images.shape[0], 1, 28, 28)))[idx] / 255.
    Y_test = test_labels[idx]
    net = Sequentual()
    net.load_model(path='/Users/oswin/Documents/BS/Test/fashion_mnist_parameters')
    y_hat = net.predict(X_test.reshape(1, 1, 28, 28))
    idx = int(y_hat.argmax(-1))

    plt.figure()
    img = X_test.reshape(28, 28) * 255.
    plt.title('标签为%s, 识别为%s, 概率为%s%%' % (label[Y_test], label[idx], export_result(y_hat[0, idx]) * 100), fontproperties=font)
    plt.imshow(img)
    plt.show()


def export_result(num):
    num_x, num_y = str(num).split('.')
    num = float(num_x + '.' + num_y[0:3])
    return num


if __name__ == '__main__':
    p.random.seed(1)
    LeNet5()
    #predict()
