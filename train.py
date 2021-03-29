import argparse
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer
import logging
logging.basicConfig(format='%(asctime)s-%(levelname)s-%(message)s', level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config/vgg-seq2seq.yml", help='config path ')
    # parser.add_argument('--checkpoint', required=False, help='your checkpoint')

    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    config = Cfg.load_config_from_file(args.config, download_base=False)
    logger.info("Loaded config from {}".format(args.config))
    # print('-- CONFIG --')
    print(config.pretty_text())
    print(config)
    trainer = Trainer(config)

    trainer.train()

if __name__ == '__main__':
    # python train.py  --config
    main()
