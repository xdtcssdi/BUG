from BUG.Model.LinearModel import LSTM_model
from BUG.function.util import load_coco_data, minibatch

if __name__ == '__main__':
    data = load_coco_data()
    word_to_idx = data['word_to_idx']
    hidden_dim = 512

    net = LSTM_model(hidden_dim=hidden_dim, word_to_idx=word_to_idx)
    net.fit(data, learning_rate=5e-3, batch_size=250, iterator=40)
