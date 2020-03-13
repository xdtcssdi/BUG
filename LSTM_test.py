from BUG.Model.model import LSTM_model
from BUG.function.util import load_coco_data, minibatch
from BUG.load_package import p


def coco():
    p.random.seed(1)
    data = load_coco_data(base_dir='/Users/oswin/datasets/coco2014', max_train=10000)
    word_to_idx = data['word_to_idx']

    hidden_dim = 512
    net = LSTM_model(hidden_dim=hidden_dim, word_to_idx=word_to_idx,
                     point=[word_to_idx['<START>'], word_to_idx['<END>'], word_to_idx['<NULL>']])
    net.fit(data, learning_rate=5e-3, batch_size=1024, iterator=100, save_epoch=1)


def predict():
    p.random.seed(1)
    data = load_coco_data(base_dir='/Users/oswin/datasets/coco2014', max_train=10000)

    word_to_idx = data['word_to_idx']

    hidden_dim = 512
    net = LSTM_model(hidden_dim=hidden_dim, word_to_idx=word_to_idx,
                     point=[word_to_idx['<START>'], word_to_idx['<END>'], word_to_idx['<NULL>']])
    net.load_model(path='/Users/oswin/Documents/BS/Test/coco_data', filename='train_params')
    net.predict(data)


if __name__ == '__main__':
    coco()


