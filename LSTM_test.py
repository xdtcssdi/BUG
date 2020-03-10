from BUG.Model.model import LSTM_model
from BUG.function.util import load_coco_data, minibatch
from BUG.load_package import p

if __name__ == '__main__':
    p.random.seed(1)
    data = load_coco_data(base_dir='/Users/oswin/datasets/coco2014', max_train=100)
    word_to_idx = data['word_to_idx']
    print(word_to_idx.keys())
    hidden_dim = 512
    net = LSTM_model(hidden_dim=hidden_dim, word_to_idx=word_to_idx,
                     point=[word_to_idx['<START>'], word_to_idx['<END>'], word_to_idx['<NULL>']])
    net.fit(data, learning_rate=5e-3, batch_size=25, iterator=100, save_epoch=10)

