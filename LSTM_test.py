from BUG.Model.model import LSTM_model
from BUG.function.util import load_coco_data, minibatch, decode_captions
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
    gt_captions, gt_captions_out, features, urls = minibatch(data, split='val', batch_size=2)[0]

    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = net.sample(features, net.layers)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        print(url)
        print('\n%s\nGT:%s' % (sample_caption, gt_caption))


if __name__ == '__main__':
    coco()
