from BUG.Model.model import Char_RNN
from BUG.function.util import load_data_gem_lyrics
from BUG.load_package import p

if __name__ == '__main__':
    p.random.seed(1)
    data, char_to_idx, idx_to_char, vocab_size = load_data_gem_lyrics()

    hidden_dim = 20
    model = Char_RNN(hidden_dim, vocab_size, char_to_idx, idx_to_char, cell='rnn')
    model.compile(optimize='Adam', learning_rate=0.005)
    model.fit(data, batch_size=64, num_steps=35, save_epoch=10, path='gem_params')
