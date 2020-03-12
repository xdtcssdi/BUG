import numpy as np

from BUG.Layers.Layer import SimpleRNN
from BUG.function.Loss import SoftCategoricalCross_entropy
from BUG.function.Optimize import Adam
from BUG.function.util import load_data_gem_lyrics, minibatch_list, one_hot
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
    n_a = net.parameters['Waa'].shape[1]  # n_a

    x = p.zeros((vocab_size, 1))
    indices = []
    if X:
        first_idx = net.char_to_ix[X[0]]
        x[first_idx, 0] = 1
        indices.append(first_idx)
    a_prev = p.zeros((n_a, 1))

    idx = -1
    counter = 0
    newline_character = net.char_to_ix['\n']

    while counter < 100:
        a = p.tanh(p.dot(net.parameters['Waa'], a_prev) + p.dot(net.parameters['Wax'], x) + net.parameters['b'])
        z = p.dot(net.parameters['Wya'], a) + net.parameters['by']
        y = net.softmax(z)
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
    batch_size = 10
    time_steps = 30
    net = SimpleRNN(vocab_size, vocab_size, idx_to_char, char_to_idx, n_a=n_a)
    loss = get_initial_loss(vocab_size, length)
    curr_loss = 0
    opt = Adam([net])
    a = p.zeros((n_a, batch_size))

    data = list(minibatch_list(example, char_to_idx, batch_size))

    lo = SoftCategoricalCross_entropy()
    for j in range(num_iterator):
        for X, Y in data:
            a, y_hat = net.forward(one_hot(X, vocab_size), a)
            curr_loss = lo.forward(Y, y_hat)
            dout = lo.backward(Y, y_hat)
            da_next = net.backward(dout)
            opt.update(learning_rate=0.005, t=j + 1)
        loss = smooth(loss, curr_loss)

        if (j+1) % 10 == 0:
            print('%s iterator loss = %s' % (j, loss))
            seed = 0
            indices = sample(net, [ch for ch in "差不多"], seed)
            print_sample(indices, idx_to_char)
            seed += 1