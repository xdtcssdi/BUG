import zipfile
from random import shuffle

import numpy as np

from BUG.Layers.Layer import SimpleRNN
from BUG.load_package import p


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0 / vocab_size) * seq_length


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt,), end='')


def sample(net, X, seed):
    vocab_size = net.by.shape[0]  # n_x
    n_a = net.Waa.shape[1]  # n_a

    x = p.zeros((vocab_size, 1))
    indices = []
    for ch in X:
        x[net.char_to_ix[ch], 0] = 1
        indices.append(net.char_to_ix[ch])
    a_prev = p.zeros((n_a, 1))

    idx = -1
    counter = 0
    newline_character = net.char_to_ix['\n']

    while counter != 100:
        a = p.tanh(p.dot(net.Waa, a_prev) + p.dot(net.Wax, x) + net.b)
        z = p.dot(net.Wya, a) + net.by
        y = net.softmax(z)

        p.random.seed(counter + seed)
        idx = p.random.choice(list(range(vocab_size)), p=y.ravel())
        indices.append(idx)

        x = y  # 传给下一个cell的输入
        x[idx] = 1  # 标记idx位置为选择的字符，传给下一cell
        a_prev = a

        seed += 1
        counter += 1
    if counter == 50:
        indices.append(net.char_to_ix['\n'])

    return indices


def rnn():
    np.random.seed(1)
    with open('/Users/oswin/Documents/BS/BUG/datasets/dinos.txt') as f:
        data = f.readlines()
        example = [seq.lower().strip() for seq in data]
    with open('/Users/oswin/Documents/BS/BUG/datasets/dinos.txt') as f:
        L = list(set(f.read().lower()))
        L.sort()
        voca_size = len(L)
        char_to_ix = {ch: i for i, ch in enumerate(L)}
        ix_to_char = {i: ch for i, ch in enumerate(L)}
    num_iterator = 1000000

    net = SimpleRNN(voca_size, voca_size, ix_to_char, char_to_ix)
    shuffle(example)
    loss = get_initial_loss(voca_size, 7)
    for j in range(num_iterator):
        idx = j % len(example)
        X = [None, ] + [char_to_ix[ch] for ch in example[idx]]
        Y = X[1:] + [char_to_ix['\n']]
        curr_loss, y_hat = net.forward(X, Y)

        da_next = net.backward(None)
        net.clip()
        net.update_params()
        loss = smooth(loss, curr_loss)

        if j % 1000 == 0:
            print('%s iterator loss = %s' % (j, loss))
            seed = 0
            for i in range(7):
                indices = sample(net, X, seed)
                print_sample(indices, ix_to_char)
                seed += 1


def load_data_jay_lyrics():
    """Load the Jay Chou lyric data set (available in the Chinese book)."""
    with zipfile.ZipFile('/Users/oswin/Documents/BS/BUG/datasets/gem_lyrics.zip') as zin:
        with zin.open('gem_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')

    idx_to_char = list(set(corpus_chars))
    # print(idx_to_char)
    char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
    # print(char_to_idx)
    vocab_size = len(char_to_idx)

    example = corpus_chars.split('\n')
    example = [seq.strip() for seq in example]

    return example, char_to_idx, idx_to_char, vocab_size


if __name__ == '__main__':
    example, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics()
    num_iterator = 1000000
    length = 50
    net = SimpleRNN(vocab_size, vocab_size, idx_to_char, char_to_idx)
    shuffle(example)
    loss = get_initial_loss(vocab_size, length)
    for j in range(num_iterator):
        idx = j % len(example)
        X = [None, ] + [char_to_idx[ch] for ch in example[idx]]
        Y = X[1:] + [char_to_idx['\n']]

        curr_loss, y_hat = net.forward(X, Y)

        da_next = net.backward(None)
        net.clip()
        net.update_params()
        loss = smooth(loss, curr_loss)

        if j % 1000 == 0:
            print('%s iterator loss = %s' % (j, loss))
            seed = 0
            for i in range(5):
                indices = sample(net, [ch for ch in '我'], seed)
                print_sample(indices, idx_to_char)
                seed += 1
