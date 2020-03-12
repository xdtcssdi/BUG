from BUG.Model.model import RNN_model
from BUG.function.util import load_poetry

if __name__ == '__main__':
    poetry_vectors, ix_to_word, word_to_ix = load_poetry(1000)

    net = RNN_model(256, word_to_ix)
    net.fit(poetry_vectors, batch_size=16)