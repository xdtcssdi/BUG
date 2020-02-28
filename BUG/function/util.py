import os
import pickle

import h5py
import numpy as np

from BUG.Layers.Layer import RNN
from BUG.function.Activation import ac_get
from BUG.load_package import p


def load_data(path):
    train_dataset = h5py.File(path[0], "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(path[1], "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    try:
        return p.asarray(train_set_x_orig), p.asarray(train_set_y_orig), p.asarray(test_set_x_orig), p.asarray(
            test_set_y_orig), classes
    except:
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='iso-8859-1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    try:
        return p.asarray(Xtr), p.asarray(Ytr), p.asarray(Xte), p.asarray(Yte)
    except:
        return Xtr, Ytr, Xte, Yte


def one_hot(labels, nb_classes=None):
    try:
        numpy_labels = p.asnumpy(labels)
        classes = np.unique(numpy_labels)
        if nb_classes is None:
            nb_classes = classes.size
        one_hot_labels = np.zeros((numpy_labels.shape[0], nb_classes))

        for i, c in enumerate(classes):
            one_hot_labels[numpy_labels == c, i] = 1.
        return p.asarray(one_hot_labels)
    except:
        classes = np.unique(labels)
        if nb_classes is None:
            nb_classes = classes.size
        one_hot_labels = np.zeros((labels.shape[0], nb_classes))

        for i, c in enumerate(classes):
            one_hot_labels[labels == c, i] = 1.
        return one_hot_labels


def unhot(one_hot_labels):
    return np.argmax(one_hot_labels, axis=-1)


def words_between_idx(doc):
    chars = list(set(doc))
    chars.sort()
    return {ch: i for i, ch in enumerate(chars)}, {i: ch for i, ch in enumerate(chars)}


def sample(layer, words_between_idx, max_chars=50, seed=0):
    if not isinstance(layer, RNN):
        raise AttributeError
    n_a, n_x = layer.Wax.shape
    a_prev = p.zeros((n_a, 1))
    x = p.zeros_like((n_x, 1))

    count = 0
    idx = -1
    indices = []
    newschar = words_between_idx['\n']
    while newschar != idx and count != max_chars:
        p.random.seed(count+seed)
        a_next = p.tanh(p.dot(layer.Waa, a_prev) + p.dot(layer.Wax, x) + layer.b)
        y = ac_get(p.dot(layer.Wya, a_next) + layer.by, 'softmax')  # 每个字符的概率
        idx = p.random.choice(list(range(n_x)), p=y.ravel())
        indices.append(idx)
        x = y
        x[idx] = 1
        a_prev = a_next

        count += 1
        seed += 1
    return indices