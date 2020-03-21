from BUG.Model.model import RNN_model
from BUG.function.util import load_data_gem_lyrics

if __name__ == '__main__':
    data, char_to_idx, idx_to_char, vocab_size = load_data_gem_lyrics()

    hidden_dim = 256
    model = RNN_model(hidden_dim, vocab_size, char_to_idx, idx_to_char, cell='lstm')
    model.compile('Adam', learning_rate=0.005)
    model.fit(data, batch_size=32, num_steps=35, save_epoch=1)
