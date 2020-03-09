from BUG.Model.model import LSTM_model
from BUG.function.util import load_coco_data, minibatch

if __name__ == '__main__':
    # data = load_coco_data(max_train=100)
    #
    # word_to_idx = data['word_to_idx']
    # hidden_dim = 512
    #
    # net = LSTM_model(hidden_dim=hidden_dim, word_to_idx=word_to_idx)
    # net.fit(data, learning_rate=5e-3, batch_size=25, iterator=40)
    a = 'aaaaaa'
    print(a.split('_')[0])
