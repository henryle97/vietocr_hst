from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def main():
    config_path = './config/vgg-seq2seq.yml'
    img_path = ''
    config = Cfg.load_config_from_file(config_path, download_base=False)

    detector = Predictor(config)

    img = Image.open(img_path)
    s = detector.predict(img)

    print(s)


if __name__ == '__main__':
    main()
