from BUG.Layers.Layer import RNN
from BUG.Model.Model import Model
from BUG.function.util import words_between_idx
from BUG.load_package import p


def strArray2intArray(chars_size, data):
    # data 字符数组
    m = len(data)
    X = p.zeros((chars_size, m))
    for i in range(m):
        str = data[i]
        for c in str:
            idx = word_to_idx[c]
            X[idx, i] = 1
    print(X)
    return X


if __name__ == '__main__':
    with open(r'/dinos.txt') as f:
        data = f.read()
    word_to_idx, idx_to_word = words_between_idx(data)
    n_x = len(word_to_idx)

    example = data.split('\n')
    strArray2intArray(n_x, example)
    # print(example)
    # m = len(example)
    # x = p.zeros((n_x , m, ))
    # num_iterator = 2500
    # for j in range(num_iterator):
    #     X = [None] + [word_to_idx[ch] for ch in example[index]]
    #     Y = X[1:] + [word_to_idx['\n']]
    #     print(X)
    #     print(Y)

    # net = Model()
    # net.add(RNN())
    # net.fit()