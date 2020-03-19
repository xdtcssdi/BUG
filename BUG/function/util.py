import json
import os
import pickle
import re
import zipfile

import h5py
import numpy as np
import requests
from bs4 import BeautifulSoup
from keras.datasets import mnist

from BUG.function.zhtools.langconv import Converter
from BUG.load_package import p


def load_data_cat(path):
    train_dataset = h5py.File(path[0], "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File(path[1], "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape(train_set_y_orig.shape[0])
    test_set_y_orig = test_set_y_orig.reshape(test_set_y_orig.shape[0])
    try:
        return p.asarray(train_set_x_orig), p.asarray(train_set_y_orig), p.asarray(test_set_x_orig), p.asarray(
            test_set_y_orig), classes
    except:
        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='iso-8859-1')
        X = datadict['mnist_dnn_parameters']
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


def one_hot(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def lyric_download():
    '''
    根据歌手id下载歌词
    :return:
    '''

    def download_by_music_id(music_id):
        # 根据歌词id下载
        url = 'http://music.163.com/api/song/lyric?' + 'id=' + str(music_id) + '&lv=1&kv=1&tv=-1'
        r = requests.get(url)
        json_obj = r.text

        j = json.loads(json_obj)
        lrc = j['lrc']['lyric']
        pat1 = re.compile(r'\[.*\]')  # 这里几行代码是把歌词中的空格和符号之类的去掉
        lrc = re.sub(pat1, '', lrc)
        pat2 = re.compile(r'.*\:.*')
        lrc = re.sub(pat2, '', lrc)
        pat3 = re.compile(r'.*\/.*')
        lrc = re.sub(pat3, '', lrc)
        lrc = lrc.strip()
        return lrc

    def get_music_ids_by_musican_id(singer_id):  # 通过一个歌手id下载这个歌手的所有歌词
        singer_url = 'http://music.163.com/artist?' + 'id=' + str(singer_id)
        r = requests.get(singer_url).text
        soupObj = BeautifulSoup(r, 'lxml')
        song_ids = soupObj.find('textarea').text
        jobj = json.loads(song_ids)

        ids = {}
        for item in jobj:
            ids[item['name']] = item['id']
        return ids

    def download_lyric(uid):
        music_ids = get_music_ids_by_musican_id(uid)
        for key in music_ids:
            try:
                text = download_by_music_id(music_ids[key])
                with open('%s.txt' % singer_id, 'a', encoding='utf-8') as f:
                    f.write('\n')
                    for t in text:
                        f.write(t)
            except:
                print('')

    print("请输入歌手的id：")
    singer_id = input()
    download_lyric(singer_id)


def load_coco_data(base_dir='/content/sample_data/coco_captioning/',
                   max_train=None,
                   pca_features=True):
    data = {}
    caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
    else:
        train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
    with h5py.File(train_feat_file, 'r') as f:
        data['train_features'] = np.asarray(f['features'])
    if pca_features:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
    else:
        val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
    with h5py.File(val_feat_file, 'r') as f:
        data['val_features'] = np.asarray(f['features'])

    dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls

    # Maybe subsample the training mnist_dnn_parameters
    if max_train is not None:
        num_train = data['train_captions'].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data['train_captions'] = data['train_captions'][mask]
        data['train_image_idxs'] = data['train_image_idxs'][mask]

    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[int(captions[i, t])]
            if word != '<NULL>':
                words.append(word)
            if word == '<END>':
                break
        decoded.append(' '.join(words))
    if singleton:
        decoded = decoded[0]
    return decoded


def minibatch(data, batch_size=100, split='train'):
    split_size = data['%s_captions' % split].shape[0]
    batch = []
    for i in range(split_size // batch_size):
        captions = data['%s_captions' % split][i * batch_size:(i + 1) * batch_size]
        image_idxs = data['%s_image_idxs' % split][i * batch_size:(i + 1) * batch_size]
        image_features = data['%s_features' % split][image_idxs]
        urls = data['%s_urls' % split][image_idxs]

        batch.append((p.asarray(captions[:, :-1]), p.asarray(captions[:, 1:]), p.asarray(image_features), urls))
    return batch


def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    classes = 10
    return X_train, y_train, X_test, y_test, classes


def load_poetry():
    with open('/Users/oswin/Documents/BS/BUG/datasets/poetry.txt', "r", encoding='utf-8') as f:
        data = f.readlines()

    content = ''.join([line.split(':')[1] for line in data])

    corpus_chars = content.replace('\n', ' ').replace('\r', ' ').replace('，', ' ').replace('。', ' ').replace('_', '')

    corpus_chars = Converter('zh-hans').convert(corpus_chars[:10000])

    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def load_data_jay_lyrics():
    with zipfile.ZipFile('/Users/oswin/Documents/BS/BUG/datasets/jaychou_lyrics.zip') as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = Converter('zh-hans').convert(f.read().decode('utf-8'))
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def load_data_gem_lyrics():
    with zipfile.ZipFile('/Users/oswin/Documents/BS/BUG/datasets/gem_lyrics.zip') as zin:
        with zin.open('gem_lyrics.txt') as f:
            corpus_chars = Converter('zh-hans').convert(f.read().decode('utf-8'))
    # corpus_chars = re.sub('[0-9a-zA-Z]', '', corpus_chars)
    corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices, char_to_idx, idx_to_char, vocab_size


def data_iter_consecutive(corpus_indices, batch_size, num_steps):
    corpus_indices = np.array(corpus_indices)
    data_len = len(corpus_indices)
    batch_len = data_len // batch_size
    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))
    epoch_size = (batch_len - 1) // num_steps
    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y
