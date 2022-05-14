import tensorflow as tf
import math
from tensorflow.keras.callbacks import Callback

from kaggle.common.logManager import LogManager
from kaggle.common.params import PARAM

TASK_NAME = "CALLBACK"
logger = LogManager().get_logger(TASK_NAME, PARAM.LOG_COMMON_FILE)


class CustomCallback(Callback):
    def __init__(self, name, epoch, train_size, batch_size, val_split=0):
        super().__init__()
        self.name = name
        # self.previout_loss = None
        self.epoch = epoch
        self.train_size = train_size
        self.batch_size = batch_size
        self.val_split = val_split
        self.train_batch_num = math.ceil(math.ceil(self.train_size * (1 - val_split) / batch_size))

    # epoch와 batch index는 0부터 시작하기 때문에 쉽게 알아보기 위하여 1부터 시작하도록 +1
    def on_epoch_begin(self, epoch, logs=None):
        logger.info('{}/{} FROM {} : EPOCH {} start'.format(epoch + 1, self.epoch, self.name, epoch + 1))

    def on_epoch_end(self, epoch, logs=None):
        logger.info('{}/{} FROM {} : EPOCH {} end'.format(epoch + 1, self.epoch, self.name, epoch + 1))

    def on_train_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            logger.info('{}/{} FROM {} : BATCH {} end'.format(batch + 1, self.train_batch_num, self.name, batch + 1))
