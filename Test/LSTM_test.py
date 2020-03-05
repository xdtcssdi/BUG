import numpy as np

from BUG.Layers.Layer import LSTM
from BUG.function.Activation import ac_get
from BUG.function.Loss import SoftCategoricalCross_entropy
from BUG.function.Optimize import Adam
from BUG.function.util import load_data_gem_lyrics, data_iter_consecutive
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
    vocab_size = net.parameters['by'].shape[0]  # n_x
    n_a = net.parameters['Wy'].shape[1]  # n_a

    x = p.zeros((vocab_size, 1))
    indices = []
    if X:
        first_idx = net.char_to_ix[X[0]]
        x[first_idx, 0] = 1
        indices.append(first_idx)
    a_prev = p.zeros((n_a, 1))
    c_prev = p.zeros_like(a_prev)
    idx = -1
    counter = 0
    newline_character = net.char_to_ix['\n']

    while counter < 100:
        merge = p.concatenate([a_prev, x], axis=0)
        f = ac_get(p.dot(net.parameters['Wf'], merge) + net.parameters['bf'], 'sigmoid')
        i = ac_get(p.dot(net.parameters['Wi'], merge) + net.parameters['bi'], 'sigmoid')
        c_hat = p.tanh(p.dot(net.parameters['Wc'], merge) + net.parameters['bc'])
        c = f * c_prev + i * c_hat
        o = ac_get(p.dot(net.parameters['Wo'], merge) + net.parameters['bo'], 'sigmoid')
        a = o * p.tanh(c)
        y = net.softmax(p.dot(net.parameters['Wy'], a) + net.parameters['by'])
        p.random.seed(counter + seed)
        idx = p.random.choice(list(range(vocab_size)), p=y.ravel())

        if X and counter < len(X) - 1:
            x.fill(0)
            idx = net.char_to_ix[X[counter + 1]]
        else:
            x = y  # 传给下一个cell的输入
        indices.append(idx)
        x[idx] = 1  # 标记idx位置为选择的字符，传给下一cell
        a_prev = a

        seed += 1
        counter += 1
    if counter == 100:
        indices.append(net.char_to_ix['\n'])

    return indices


if __name__ == '__main__':
    p.random.seed(1)
    example, char_to_idx, idx_to_char, vocab_size = load_data_gem_lyrics()
    num_iterator = 1000000
    length = 50
    n_a = 50
    batch_size = 300
    time_steps = 30
    net = LSTM(n_x=vocab_size, n_y=vocab_size, ix_to_char=idx_to_char, char_to_ix=char_to_idx, n_a=50)
    loss = get_initial_loss(vocab_size, length)
    curr_loss = 0
    opt = Adam([net])
    a = p.zeros((n_a, batch_size))
    data = list(data_iter_consecutive(example, batch_size, time_steps, vocab_size))
    lo = SoftCategoricalCross_entropy()
    for j in range(num_iterator):
        for X, Y in data:
            X = X.transpose(1, 0, 2).transpose(2, 1, 0)
            Y = Y.transpose(1, 0, 2).transpose(2, 1, 0)
            a, y_hat, c = net.forward(X, a)
            curr_loss = lo.forward(Y, y_hat)
            dout = lo.backward(Y, y_hat)
            da_next = net.backward(dout)
            opt.update(learning_rate=0.005, t=j + 1)

        loss = smooth(loss, curr_loss)
        if j % 10 == 0:
            print('%s iterator loss = %s' % (j, loss))
            seed = 0
            indices = sample(net, [ch for ch in "差不多"], seed)
            print_sample(indices, idx_to_char)
            seed += 1
