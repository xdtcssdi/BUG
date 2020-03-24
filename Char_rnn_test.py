from BUG.Model.model import Char_RNN
from BUG.function.util import load_data_gem_lyrics, load_data_ana
from BUG.load_package import p
import matplotlib.pyplot as plt


def show_loss(costs, perplexity):
    x_axis = len(costs)
    plt.title('Result Analysis')
    plt.plot(range(x_axis), costs, color='green', label='costs: ' + str(costs[-1]))
    plt.plot(range(x_axis), perplexity, color='red', label='perplexity: ' + str(perplexity[-1]))
    plt.legend()  # 显示图例

    plt.xlabel('iteration times')
    plt.ylabel('rate')
    plt.show()


if __name__ == '__main__':
    p.random.seed(1)
    data, char_to_idx, idx_to_char, vocab_size = load_data_gem_lyrics()

    hidden_dim = 20
    model = Char_RNN(hidden_dim, vocab_size, char_to_idx, idx_to_char, 300, cell='rnn')
    model.compile(optimize='Adam', learning_rate=0.005)
    model.fit(data, batch_size=64, num_steps=35, save_epoch=10, path='gem_params')
    show_loss(model.costs, model.perplexity)

