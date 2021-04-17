import argparse
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def main():
    config_path = './logs/hw_word_seq2seq/config.yml'
    img_path = './logs/hw_word_seq2seq_test/4e788cee-d0f1-4805-894f-d72a9f9286a1_1.jpg'
    config = Cfg.load_config_from_file(config_path, download_base=False)

    config['weights'] = './logs/hw_word_seq2seq/best.pt'

    print(config.pretty_text())

    detector = Predictor(config)

    img = Image.open(img_path)
    s = detector.predict(img)

    print(s)


if __name__ == '__main__':
    main()
