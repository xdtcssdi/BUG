import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from BUG.Layers.Layer import Dense
from BUG.Model.model import Sequentual
from BUG.function import Loss
from BUG.function.evaluate import evaluate_many
from BUG.function.util import load_mnist

np.set_printoptions(threshold=np.inf, suppress=True)


def mnist():
    # 数据预处理
    X_train, y_train, X_test, y_test, classes = load_mnist()

    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32)[:60000] / 255.
    Y_train = y_train[:60000]

    X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255.
    y_test = y_test

    accuracy = evaluate_many
    # 创建网络架构
    net = Sequentual()
    net.add(Dense(256, activation='relu', batchNormal=True))
    net.add(Dense(classes, activation="softmax"))
    net.compile(lossMode=Loss.SoftCategoricalCross_entropy(), optimize='Adam', accuracy=evaluate_many)
    net.fit(X_train, Y_train, X_test, y_test, batch_size=1024, learning_rate=0.001, save_epoch=10,
            path='mnist_parameters',
            iterator=20)


def pre_pic(picName):
    # 先打开传入的原始图片
    img = Image.open(picName)
    # 使用消除锯齿的方法resize图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)

    # 变成灰度图，转换成矩阵
    im_arr = np.array(reIm.convert("L"))

    threshold = 30
    # 对图像进行二值化处理
    tmp_im = 255 - im_arr
    mask = tmp_im > threshold
    im_arr *= mask
    img = im_arr
    # reshape
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)

    return img_ready, img


def predict():
    data, img = pre_pic("/Users/oswin/Documents/BS/Test/test_data/img4.png")

    net = Sequentual()
    net.load_model(path='mnist_parameters')
    y_hat = net.predict(data.reshape(1, -1))
    idx = y_hat.argmax(-1)
    print('识别为： %d, 概率为%.2f%%' % (idx, y_hat[0, idx] * 100))
    plt.figure()
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    np.random.seed(1)
    mnist()
    #predict()
