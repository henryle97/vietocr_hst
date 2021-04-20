import argparse
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os
import time

# python predict_seq2seq.py --config logs/hw_word_seq2seq_finetuning_170k_v2/config.yml --weight logs/hw_word_seq2seq_finetuning_170k_v2/best.pt --img image

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./logs/hw_word_seq2seq/config.yml')
    parser.add_argument('--weight', type=str, default='./logs/hw_word_seq2seq/best.pt')
    parser.add_argument('--img', type=str, default=None, required=True)
    args = parser.parse_args()

    config = Cfg.load_config_from_file(args.config, download_base=False)

    config['weights']= args.weight

    print(config.pretty_text())

    detector = Predictor(config)
    if os.path.isdir(args.img):
        img_paths = os.listdir(args.img)
        for img_path in img_paths:
            try:
                img = Image.open(args.img + '/' + img_path)
            except:
                 continue
            t1 = time.time()
            s, prob = detector.predict(img, return_prob=True)
            print('Text in {} is:\t {} | prob: {:.2f} | times: {:.2f}'.format(img_path, s, prob, time.time() - t1))
    else:
        t1 = time.time()
        img = Image.open(args.img)
        s, prob = detector.predict(img, return_prob=True)
        print('Text in {} is:\t {} | prob: {:.2f} | times: {:.2f}'.format(args.img, s, prob, time.time() - t1))

def predict_file():
    config_path = './logs/hw_word_seq2seq/config.yml'
    config = Cfg.load_config_from_file(config_path, download_base=False)

    config['weights']= './logs/hw_word_seq2seq_finetuning/best.pt'

    print(config.pretty_text())

    detector = Predictor(config)

    detector.gen_annotations('./DATA/data_verifier/hw_word_15k_labels.txt', './DATA/data_verifier/hw_word_15k_labels_preds.txt', data_root='./DATA/data_verifier')

if __name__ == '__main__':
    main()
    # predict_file()
