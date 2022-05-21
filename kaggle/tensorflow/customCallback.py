import tensorflow as tf
import math
from tensorflow.keras.callbacks import Callback

from kaggle.common.logManager import LogManager
from kaggle.common.params import PARAM

TASK_NAME = "CALLBACK"
logger = LogManager().get_logger(TASK_NAME, PARAM.LOG_COMMON_FILE)


class CustomCallback(Callback):
    def __init__(self, name, epoch, train_size, batch_size, val_split=0, metrics=None):
        super().__init__()
        if metrics is None:
            metrics = []
        self.name = name
        # self.previout_loss = None
        self.epoch = epoch
        self.train_size = train_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.train_batch_num = math.ceil(math.ceil(self.train_size * (1 - val_split) / batch_size))
        self.metrics = metrics

    # epoch와 batch index는 0부터 시작하기 때문에 쉽게 알아보기 위하여 1부터 시작하도록 +1
    def on_epoch_begin(self, epoch, logs=None):
        logger.info('FROM {} : EPOCH {}/{} start'.format(self.name, epoch + 1, self.epoch))

    def on_epoch_end(self, epoch, logs=None):
        if 'accuracy' in self.metrics:
            logger.info(
                'FROM {} : loss = {:.5f}, val_loss = {:.5f}, acc = {:.5f}, val_acc = {:.5f}'
                    .format(self.name,
                            logs['loss'],
                            logs['val_loss'],
                            logs['accuracy'],
                            logs['val_accuracy']))
        else:
            logger.info('FROM {} : loss = {:.5f}, val_loss = {:.5f}'.format(self.name, logs['loss'], logs['val_loss']))
        logger.info('FROM {} : EPOCH {}/{} end'.format(self.name, epoch + 1, self.epoch))

    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            logger.info('FROM {} : BATCH {}/{} end'.format(self.name, batch + 1, self.train_batch_num))

    def on_train_begin(self, logs=None):
        logger.info('Model train start.')

    def on_train_end(self, logs=None):
        logger.info('Model train end.')
