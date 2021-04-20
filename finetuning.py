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
    dataset_params = {
        'name': 'hw_word',
        'data_root': './DATA',
        'is_padding': True,
        'image_max_width': 100,
        'train_lmdb': ['train_hw_word', 'hw_word_9k_good', 'hw_word_50k_dict_3k', 'valid_hw_word', 'hw_word_70k_dict_full_filter'],
        'valid_lmdb': 'test_hw_word'
    }
    config['monitor']['log_dir'] = './logs/hw_word_seq2seq_finetuning_240k'

    trainer_params = {
        'batch_size': 32,
        'print_every': 200,
        'valid_every': 5 * 200,
        'iters': 150000,
        'metrics': 5000,
        'pretrained': './logs/hw_word_seq2seq_finetuning_170k_v2/best.pt',
        'resume_from': None,
        'is_finetuning': False
    }

    config['aug']['masked_language_model'] = False

    optim_params = {
        'max_lr': 0.00001
    }
    config['optimizer'].update(optim_params)

    config['trainer'].update(trainer_params)
    # config['trainer']['resume_from'] = './logs/hw_small_finetuning/last.pt'
    config['dataset'].update(dataset_params)



    print(config.pretty_text())
    # print(config)
    trainer = Trainer(config, pretrained=False)
    # trainer.visualize_dataset()
    trainer.train()

if __name__ == '__main__':
    # python train.py  --config
    main()
