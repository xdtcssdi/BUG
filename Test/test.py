import collections

from BUG.Layers.Layer import SimpleRNN
from BUG.function.Loss import SoftCategoricalCross_entropy
from BUG.function.Optimize import Adam
from BUG.function.util import one_hot
from BUG.load_package import p


def load_poetry():
    poetry_file = '/Users/oswin/Documents/BS/BUG/datasets/poetry.txt'

    # 诗集
    poetrys = []
    with open(poetry_file, "r", encoding='utf-8', ) as f:
        for line in f:
            # print line
            try:
                title, content = line.strip().split(':')
                content = content.replace(' ', '')
                if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                    continue
                if len(content) < 5 or len(content) > 79:
                    continue
                content = '[' + content + ']'
                poetrys.append(content)
            except Exception as e:
                pass

    # 按诗的字数排序
    poetrys = sorted(poetrys, key=lambda line: len(line))

    print(u"唐诗总数: ")
    print(len(poetrys))

    # 统计每个字出现次数
    all_words = []
    for poetry in poetrys:
        all_words += [word for word in poetry]

    counter = collections.Counter(all_words)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])

    words, _ = zip(*count_pairs)

    # 取前多少个常用字
    words = words[:len(words)] + (' ',)

    # 每个字映射为一个数字ID
    word_num_map = dict(zip(words, range(len(words))))

    # 把诗转换为向量形式，参考TensorFlow练习1
    char_to_idx = lambda word: word_num_map.get(word, len(words))

    poetrys_vector = [list(map(char_to_idx, poetry)) for poetry in poetrys]

    idx_to_idx = {value: key for key, value in word_num_map.items()}
    return word_num_map, idx_to_idx, poetrys_vector


def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001


def get_initial_loss(vocab_size, seq_length):
    return -p.log(1.0 / vocab_size) * seq_length


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


def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character
    print('%s' % (txt,), end='')


if __name__ == '__main__':
    char_to_idx, idx_to_idx, poetrys_vector = load_poetry()
    batch_size = 64
    n_chunk = len(poetrys_vector) // batch_size
    x_batches = []
    y_batches = []
    for i in range(n_chunk):
        start_index = i * batch_size
        end_index = start_index + batch_size

        batches = poetrys_vector[start_index:end_index]
        length = max(map(len, batches))  # batch 中最大的词长度
        xdata = p.full((batch_size, length), char_to_idx[' '], p.int32)
        for row in range(batch_size):
            xdata[row, :len(batches[row])] = batches[row]
        ydata = p.copy(xdata)
        ydata[:, :-1] = xdata[:, 1:]
        """
        xdata             ydata
        [6,2,4,6,9]       [2,4,6,9,9]
        [1,4,2,8,5]       [4,2,8,5,5]
        """
        x_batches.append(xdata)
        y_batches.append(ydata)

    n_a = 50
    length = 50
    vocab_size = len(char_to_idx)
    net = SimpleRNN(vocab_size, vocab_size, idx_to_idx, char_to_idx, n_a=n_a)
    loss = get_initial_loss(vocab_size, length)
    curr_loss = 0
    opt = Adam([net])
    lo = SoftCategoricalCross_entropy()
    num_iterator = 1000
    for i in range(num_iterator):
        for X, Y in zip(x_batches, y_batches):
            X = one_hot(X, vocab_size)
            a, y_hat = net.forward(X)
            curr_loss = lo.forward(Y, y_hat)
            dout = lo.backward(Y, y_hat)
            da_next = net.backward(dout)
            opt.update(learning_rate=0.005, t=i + 1)
            loss = smooth(loss, curr_loss)
            print(loss)

        if i % 10 == 0:
            print('%s iterator loss = %s' % (i, loss))
            seed = 0
            indices = sample(net, None, seed)
            print_sample(indices, idx_to_idx)
            seed += 1
