import os
import logging
import sys
import time

class Logger():
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        format_ = '%(asctime)s - %(levelname)s - %(message)s'
        formater = logging.Formatter(format_)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        # stream handler
        instance = logging.StreamHandler(sys.stdout)
        instance.setLevel(logging.INFO)
        instance.setFormatter(formater)
        logger.addHandler(instance)

        # file handler
        file_instance = logging.FileHandler(os.path.join(log_dir, '{}.log'.format(timestamp)), 'w')
        file_instance.setFormatter(formater)
        file_instance.setLevel(logging.INFO)
        logger.addHandler(file_instance)

        self.logger = logger

    def info(self, string):
        self.logger.info(string)


