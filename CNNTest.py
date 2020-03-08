from tensorflow import keras

from BUG.Layers.Layer import Convolution, Pooling, Dense
from BUG.Model.model import Linear_model
from BUG.function.evaluate import evaluate_many
from BUG.load_package import p


def LeNet5():
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    X_train = p.array(p.reshape(train_images, (train_images.shape[0], 1, 28, 28))) / 255.
    X_test = p.array(p.reshape(test_images, (test_images.shape[0], 1, 28, 28))) / 255.
    Y_train = train_labels
    Y_test = test_labels
    net = Linear_model()
    net.add(Convolution(filter_count=6, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='valid'))
    net.add(Convolution(filter_count=16, filter_shape=(5, 5), batchNormal=True))
    net.add(Pooling(filter_shape=(2, 2), stride=2, mode='max', paddingMode='valid'))
    net.add(Dense(120, batchNormal=True, flatten=True))
    net.add(Dense(84, batchNormal=True))
    net.add(Dense(10, batchNormal=True, activation="softmax"))
    net.compile()
    accuracy = evaluate_many
    net.fit(X_train, Y_train, accuracy, X_test, Y_test, batch_size=1024, iterator=100,
            learning_rate=0.0075, is_normalizing=False,path='/content/drive/My Drive/Colab Notebooks',
            lossMode='SoftmaxCrossEntry', optimize='Adam')


if __name__ == '__main__':
    p.random.seed(1)
    LeNet5()
