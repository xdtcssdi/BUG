import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import requests as req
from PIL import Image
from io import BytesIO
from BUG.Model.model import LSTM_model
from BUG.function.util import load_coco_data, minibatch, decode_captions
from BUG.load_package import p

font = FontProperties(fname='/Users/oswin/Documents/BS/BUG/datasets/PingFang.ttc', size=8)


def coco():
    p.random.seed(1)
    data = load_coco_data(base_dir='/Users/oswin/datasets/coco2014', max_train=10000)
    word_to_idx = data['word_to_idx']

    hidden_dim = 256
    net = LSTM_model(hidden_dim=hidden_dim, word_to_idx=word_to_idx,
                     point=[word_to_idx['<START>'], word_to_idx['<END>'], word_to_idx['<NULL>']])
    net.fit(data, learning_rate=5e-3, batch_size=1024, iterator=100, save_epoch=1, path='a', is_print=True)


def predict():
    p.random.seed(1)
    data = load_coco_data(base_dir='/Users/oswin/datasets/coco2014', max_train=10000)

    word_to_idx = data['word_to_idx']

    hidden_dim = 256
    net = LSTM_model(hidden_dim=hidden_dim, word_to_idx=word_to_idx,
                     point=[word_to_idx['<START>'], word_to_idx['<END>'], word_to_idx['<NULL>']])
    net.load_model(path='/Users/oswin/Documents/BS/Test/coco_parameters')
    gt_captions, gt_captions_out, features, urls = minibatch(data, split='val', batch_size=2)[11]

    gt_captions = decode_captions(gt_captions, data['idx_to_word'])

    sample_captions = net.sample(features, net.layers)
    sample_captions = decode_captions(sample_captions, data['idx_to_word'])

    plt.figure()
    count = 1
    for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
        plt.subplot(2, 1, count)
        count += 1
        response = req.get(url)
        image = Image.open(BytesIO(response.content))
        plt.imshow(image)
        plt.text(0, 0, '\norigin:%s\n%s' % (gt_caption, sample_caption), fontproperties=font)

    plt.show()


if __name__ == '__main__':
    coco()
    # predict()
