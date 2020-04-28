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
    X_train = p.array(p.reshape(train_images, (train_images.shape[0], 1, 28, 28))) / 255.
    X_test = p.array(p.reshape(test_images, (test_images.shape[0], 1, 28, 28))) / 255.
    Y_train = p.array(train_labels)
    Y_test = p.array(test_labels)
    net = Sequentual()
    net.add(Convolution(filter_count=6, filter_shape=(5, 5), batchNormal=True, activation='sigmoid'))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='same'))
    net.add(Convolution(filter_count=16, filter_shape=(5, 5), batchNormal=True, activation='sigmoid'))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='same'))
    net.add(Dense(120, batchNormal=True, flatten=True, activation='sigmoid'))
    net.add(Dense(84, batchNormal=True, activation='sigmoid'))
    net.add(Dense(classes, activation="sigmoid"))
    net.compile(lossMode=Loss.SoftCategoricalCross_entropy(), optimize='Adam', evaluate=evaluate_many)
    net.fit(X_train, Y_train, batch_size=64, iterator=5, learning_rate=0.001, is_print=False,
            save_epoch=100000, path='fashion_mnist_parameters1')
    _, acc = net.evaluate(X_test, Y_test)
    print('test acc = %.2f%%' % acc*100)


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
