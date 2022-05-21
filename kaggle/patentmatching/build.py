import numpy as np
import math
import pandas as pd
import tensorflow.keras.callbacks
import tensorflow_datasets as tfds
from keras.layers import Flatten
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.utils import normalize
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import CosineSimilarity, Reduction
from datetime import datetime
from kaggle.common.config import Config
from kaggle.common.modelConfigManager import ModelConfigManager
from kaggle.common.pathManager import PathManager
from kaggle.common.logManager import LogManager
from kaggle.common.params import PARAM
from kaggle.common.classificationCode import TaskMap
from kaggle.tensorflow.customCallback import CustomCallback
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# 전역 공통 디렉토리 변수 선언

server_config = Config(None, PARAM.FILE_CONFIG)
path_manager = PathManager()
model_config = ModelConfigManager()
PATH_HOME = server_config.get_param(PARAM.PARAM_PATH_HOME, "./")
TASK_NAME = TaskMap.PATENTMATCHING.name  # modelConfig.json의 TASK_NAME과 동일해야함
logger = LogManager().get_logger(TASK_NAME, PARAM.LOG_COMMON_FILE)


class FirstModel:

    def __init__(self):
        self.MODEL_TYPE = TaskMap.PATENTMATCHING.value.get('MSE')  # modelConfig.json의 MODEL_TYPE과 동일해야함
        self.model_config_dict = model_config.get_task_model_config(TASK_NAME, self.MODEL_TYPE)
        self.train_1_X = None
        self.train_2_X = None
        self.train_y = None
        self.train_size = None
        # self.train_batch_num = None
        self.train_X_len = None
        self.anchor_len = None
        self.target_len = None
        self.vocab_size = None
        self.output_dim = 512
        self.model = None
        self.tokenizer = None

        pass

    def preprocessing(self):
        logger.debug('RESOURCE_NAME = {}'.format(TASK_NAME))
        logger.info('RS_HOME = {}'.format(server_config.get_param(PARAM.PARAM_RESOURCES_HOME)))
        logger.info('RS_FOLDER = {}'.format(server_config.get_param(PARAM.PARAM_RESOURCES_FOLDER)))
        df = pd.read_csv(os.path.join(path_manager.get_data_path(TASK_NAME, self.MODEL_TYPE), "train.csv"))
        # print(df)
        # print(df['score'].unique())

        logger.debug('data = {}'.format(df))
        self.train_size = len(df)
        logger.debug('train data size = {}'.format(self.train_size))
        self.train_1_X = df['anchor']
        self.train_2_X = df['target']
        self.train_y = df['score']
        result = pd.concat([self.train_1_X, self.train_2_X])
        logger.debug('result count = {}'.format(result.count()))
        result.index = [i for i in range(0, result.count())]

        # print(result)

        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(result, target_vocab_size=10000)
        # print('tokenizer subword size = ', len(self.tokenizer.subwords))
        # self.vocab_size = len(self.tokenizer.subwords)
        logger.debug('tokenizer vocab size = {}'.format(self.tokenizer.vocab_size))
        self.vocab_size = self.tokenizer.vocab_size

        train_X = [self.tokenizer.encode(elem) for elem in result]
        print('type of train_X : ', type(train_X))
        # print(train_X)
        self.train_X_len = max([len(line) for line in train_X])

        train_X_max_val = max([max(line) for line in train_X])
        logger.debug('max_val on train_X : {}'.format(train_X_max_val))
        train_X = pad_sequences(train_X, maxlen=self.train_X_len)

        self.train_1_X = train_X[:len(df)]
        self.train_2_X = train_X[len(df):]

        logger.debug('padding 후 anchor shape : [ {} , {} ]'.format(len(self.train_1_X), len(self.train_1_X[0])))
        logger.debug('padding 후 target shape : [ {} , {} ]'.format(len(self.train_2_X), len(self.train_2_X[0])))
        logger.debug('label shape : {}'.format(self.train_y.shape))

    def makeModel(self):
        anchor = Input(shape=(self.train_X_len,))
        target = Input(shape=(self.train_X_len,))
        # Embedding층에 들어갈 입력 두개. 원핫인코딩이 되지 않은 상태이며,
        # 따라서 (batch_size, input_length) 인데 임베딩층을 거치면 (batch_size, input_length, embed_output_dim) 이 된다.

        logger.debug('model Input shape : {}, {}'.format(anchor.shape, target.shape))

        anchor_embed = Embedding(self.vocab_size, self.output_dim, mask_zero=True
                                 , input_length=self.train_X_len
                                 )
        x = anchor_embed(anchor)
        logger.debug('anchor embed shape : {}'.format(x.shape))
        anchor_lstm = LSTM(units=256)  # return_state=True이면 list형태로 3개의 텐서, result, state_c, state_h 가 반환됨.
        anchor_output = anchor_lstm(x)

        target_embed = Embedding(self.vocab_size, self.output_dim, mask_zero=True
                                 , input_length=self.train_X_len
                                 )
        y = target_embed(target)
        logger.debug('target embed shape : {}'.format(y.shape))
        target_lstm = LSTM(units=256)
        target_output = target_lstm(y)

        logger.debug('lstm output shape : {}, {}'.format(anchor_output.shape, target_output.shape))

        option = 'manual'

        cosine_sim = self.calc_cosine_sim(anchor_output, option, target_output)

        cosine_sim = (cosine_sim + 1) / 2

        self.model = Model(inputs=[anchor, target], outputs=cosine_sim)

        metrics = [metrics for metrics in self.model_config_dict.get(PARAM.PARAM_METRICS).split(',')]
        self.model.compile(optimizer=self.model_config_dict.get(PARAM.PARAM_OPTIMIZER),
                           loss=self.model_config_dict.get(PARAM.PARAM_LOSS),
                           metrics=metrics)

        modelCheckPointFile = os.path.join(
            path_manager.get_checkpoint_path(TASK_NAME, self.MODEL_TYPE,
                                             self.model_config_dict.get(PARAM.PARAM_MODEL_NAME), clear_option=True),
            self.model_config_dict.get(PARAM.PARAM_MODEL_CHECKPOINT_NAME))
        cpCallback = ModelCheckpoint(filepath=modelCheckPointFile, verbose=1, monitor='val_loss',
                                     save_weights_only=True, mode='min', save_best_only=True)

        # try:
        self.model.fit(x=[self.train_1_X, self.train_2_X], y=self.train_y,
                       batch_size=self.model_config_dict.get(PARAM.PARAM_BATCH_SIZE),
                       epochs=self.model_config_dict.get(PARAM.PARAM_EPOCHS),
                       validation_split=self.model_config_dict.get(PARAM.PARAM_VAL_SPLIT),

                       callbacks=[CustomCallback('callback', self.model_config_dict.get(PARAM.PARAM_EPOCHS),
                                                 self.train_size, self.model_config_dict.get(PARAM.PARAM_BATCH_SIZE),
                                                 self.model_config_dict.get(PARAM.PARAM_VAL_SPLIT), metrics=metrics),
                                  cpCallback]
                       )
        # except Exception as ex:
        #     print('error occured..', ex)

    def calc_cosine_sim(self, anchor_output, option, target_output):
        if option == 'useLoss':
            # anchor_norm = self.customNorm(anchor_output, axis=1)
            anchor_norm = tf.expand_dims(anchor_output, 2)
            logger.debug('anchor norm shape : {}'.format(anchor_norm.shape))

            # target_norm = self.customNorm(target_output, axis=1)
            target_norm = tf.expand_dims(anchor_output, 2)
            logger.debug('target norm shape : {}'.format(target_norm.shape))

            # 그냥 * 로는 원하는 모양으로 행렬곱이 되지 않아 matmul 사용
            # 3d shape tensor 그 이상에서는 행렬곱을 하고싶은 위치만 transpose되어야 행렬곱 가능.
            # expand_dims로 shape 조정 (복잡할때는 transpose이용하여 shape 조정 가능)

            # cosine_sim = tf.matmul(anchor_norm, target_norm)

            cosine_loss = CosineSimilarity(axis=1, reduction=Reduction.NONE)
            # cosine_loss는 -1로갈수록 유사도가 높아짐.
            cosine_sim = cosine_loss(anchor_norm, target_norm)

            logger.debug('after matmul cosine_sim shape : {}'.format(cosine_sim.shape))
            cosine_sim = tf.reshape(cosine_sim, [-1, 1])
            logger.debug('after matmul cosine_sim shape : {}'.format(cosine_sim.shape))

        elif option == 'manual':
            anchor_norm = self.customNorm(anchor_output, axis=1)
            anchor_norm = tf.expand_dims(anchor_norm, 1)
            logger.debug('anchor norm shape : {}'.format(anchor_norm.shape))

            target_norm = self.customNorm(target_output, axis=1)
            target_norm = tf.expand_dims(target_norm, 2)
            logger.debug('target norm shape : {}'.format(target_norm.shape))

            cosine_sim = tf.matmul(anchor_norm, target_norm)

            logger.debug('after matmul cosine_sim shape : {}'.format(cosine_sim.shape))
            cosine_sim = tf.reshape(cosine_sim, [-1, 1])
            logger.debug('after matmul cosine_sim shape : {}'.format(cosine_sim.shape))
        return cosine_sim

    def customNorm(self, tensor, axis=1):
        norm = tf.nn.l2_normalize(tensor, axis=axis)
        logger.debug('customNorm : after norm shape = {}'.format(norm.shape))

        return norm

    def saveModel(self):
        # format_data = "%d%m%y%H%M%S"
        # date = datetime.strftime(datetime.now(), format_data)
        # modelName = "mseModel_{}.h5".format(date)
        self.model.save(os.path.join(path_manager.get_model_path(TASK_NAME, self.MODEL_TYPE),
                                     self.model_config_dict.get(PARAM.PARAM_MODEL_NAME)))

    def loadModel(self):
        logger.debug('load model start')
        logger.debug('model name = {}'.format(self.model_config_dict.get(PARAM.PARAM_MODEL_NAME)))
        self.model = keras.models.load_model(os.path.join(path_manager.get_model_path(TASK_NAME, self.MODEL_TYPE),
                                                          self.model_config_dict.get(PARAM.PARAM_MODEL_NAME)))
        self.preprocessing()
        # df = pd.read_csv(os.path.join(server_config.get_param(PARAM.PARAM_RESOURCES_HOME),
        #                               server_config.get_param(PARAM.PARAM_RESOURCES_FOLDER),
        #                               RESOURCE_NAME, "train.csv"))
        #
        # self.train_1_X = df['anchor']
        # self.train_2_X = df['target']
        # self.train_y = df['score']
        # result = pd.concat([self.train_1_X, self.train_2_X])
        #
        # result.index = [i for i in range(0, result.count())]
        #
        # self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(result, target_vocab_size=10000)
        #
        # self.vocab_size = self.tokenizer.vocab_size
        #
        # train_X = [self.tokenizer.encode(elem) for elem in result]
        # # print(type(train_X))
        # # print(train_X)
        # self.train_X_len = max([len(line) for line in train_X])
        # train_X_max_val = max([max(line) for line in train_X])
        # logger.debug('max_val on train_X : {}'.format(train_X_max_val))
        # train_X = pad_sequences(train_X, maxlen=self.train_X_len)
        #
        # self.train_1_X = train_X[:len(df)]
        # self.train_2_X = train_X[len(df):]

    def predictPartTrainData(self):
        anchor = self.train_1_X[0:2]
        target = self.train_2_X[0:2]
        logger.debug('anchor shape : ({},{})'.format(len(anchor), len(anchor[0])))
        logger.debug('target shape : ({},{})'.format(len(target), len(target[0])))
        simScore = self.train_y[0:2]
        logger.debug('simScore : {}'.format(simScore))
        # anchor = self.tokenizer.decode(anchor)
        # anchor = pad_sequences(anchor, maxlen=self.anchor_len)
        # target = self.tokenizer.decode(target)
        # target = pad_sequences(target, maxlen=self.target_len)
        sim = self.model.predict([anchor, target])
        logger.debug('predict SimScore : {}'.format(sim))

    def predictAllTrainData(self):
        sim = self.model.predict([self.train_1_X, self.train_2_X])
        for i in range(100):
            logger.debug('predict score = {}, label = {}'.format(sim[i], self.train_y[i]))

    def predictTestData(self):
        df = pd.read_csv(os.path.join(path_manager.get_data_path(TASK_NAME, self.MODEL_TYPE), "test.csv"))
        test_id = df['id']
        anchor = df['anchor']
        target = df['target']

        test_1_X = [self.tokenizer.encode(elem) for elem in anchor]
        test_2_X = [self.tokenizer.encode(elem) for elem in target]

        test_1_X = pad_sequences(test_1_X, maxlen=self.train_X_len)
        test_2_X = pad_sequences(test_2_X, maxlen=self.train_X_len)

        # print(test_1_X)
        # print(test_2_X)

        sim = self.model.predict([test_1_X, test_2_X])
        for i in range(len(test_id)):
            logger.debug(
                'predict id ={}, anchor = {}, target = {}, score = {}'.format(test_id[i], anchor[i], target[i], sim[i]))


class Test:
    def __init__(self):
        self.embed_output_dim = 512
        self.lstm_output_dim = 2
        self.vocab_size = 12099
        pass

    def checkShape(self):
        # anchor = np.random.random((1111, 15))
        anchor = tf.random.uniform(shape=(3, 15), minval=1, maxval=self.vocab_size, dtype=tf.dtypes.int32)
        # target = np.random.random((1111, 15))
        target = tf.random.uniform(shape=(3, 15), minval=1, maxval=self.vocab_size, dtype=tf.dtypes.int32)
        # Embedding층에 들어갈 입력 두개. 원핫인코딩이 되지 않은 상태이며,
        # 따라서 (batch_size, input_length) 인데 임베딩층을 거치면 (batch_size, input_length, embed_output_dim) 이 된다.

        logger.debug("model Input shape : {}, {}".format(anchor.shape, target.shape))

        anchor_embed = Embedding(self.vocab_size, self.embed_output_dim, mask_zero=True
                                 # , input_length=self.train_X_len
                                 )
        x = anchor_embed(anchor)
        print('anchor embed shape : ', x.shape)
        anchor_lstm = LSTM(
            units=self.lstm_output_dim)  # return_state=True이면 list형태로 3개의 텐서, result, state_c, state_h 가 반환됨.
        anchor_output = anchor_lstm(x)

        target_embed = Embedding(self.vocab_size, self.embed_output_dim, mask_zero=True
                                 # , input_length=self.train_X_len
                                 )
        y = target_embed(target)
        print('target embed shape : ', y.shape)
        target_lstm = LSTM(units=self.lstm_output_dim)
        target_output = target_lstm(y)

        print('lstm output shape : ', anchor_output.shape, target_output.shape)

        option = 'manual'
        cosine_sim = self.calc_cosine_sim(anchor_output, option, target_output)

        # 3d shape tensor 그 이상에서는 행렬곱을 하고싶은 위치만 transpose되어야 행렬곱 가능.
        # expand_dims로 shape 조정 (복잡할때는 transpose이용하여 shape 조정 가능)

        # cosine_sim = tf.matmul(anchor_norm, target_norm)

        # cosine_sim = Flatten()(cosine_sim)

        print('min value : ', min(cosine_sim), ', max value : ', max(cosine_sim))
        cosine_sim = (cosine_sim + 1) / 2
        print('min value : ', min(cosine_sim), ', max value : ', max(cosine_sim))

        print('cosine sim shape : ', cosine_sim.shape)

    def calc_cosine_sim(self, anchor_output, option, target_output):
        if option == 'useLoss':
            # anchor_norm = self.customNorm(anchor_output, axis=1)
            anchor_norm = tf.expand_dims(anchor_output, 2)
            logger.debug('anchor norm shape : {}'.format(anchor_norm.shape))

            # target_norm = self.customNorm(target_output, axis=1)
            target_norm = tf.expand_dims(anchor_output, 2)
            logger.debug('target norm shape : {}'.format(target_norm.shape))

            # 그냥 * 로는 원하는 모양으로 행렬곱이 되지 않아 matmul 사용
            # 3d shape tensor 그 이상에서는 행렬곱을 하고싶은 위치만 transpose되어야 행렬곱 가능.
            # expand_dims로 shape 조정 (복잡할때는 transpose이용하여 shape 조정 가능)

            # cosine_sim = tf.matmul(anchor_norm, target_norm)

            cosine_loss = CosineSimilarity(axis=1, reduction=Reduction.NONE)
            # cosine_loss는 -1로갈수록 유사도가 높아짐.
            cosine_sim = cosine_loss(anchor_norm, target_norm)

            logger.debug('after matmul cosine_sim shape : {}'.format(cosine_sim.shape))
            cosine_sim = tf.reshape(cosine_sim, [-1, 1])
            logger.debug('after matmul cosine_sim shape : {}'.format(cosine_sim.shape))

        elif option == 'manual':
            anchor_norm = self.customNorm(anchor_output, axis=1)
            anchor_norm = tf.expand_dims(anchor_norm, 1)
            logger.debug('anchor norm shape : {}'.format(anchor_norm.shape))

            target_norm = self.customNorm(target_output, axis=1)
            target_norm = tf.expand_dims(target_norm, 2)
            logger.debug('target norm shape : {}'.format(target_norm.shape))

            print('before matmul : \n anchor_norm = {} \n target_norm = {}'.format(anchor_norm, target_norm))
            cosine_sim = tf.matmul(anchor_norm, target_norm)
            print('after matmul : \n cosine_sim = {}'.format(cosine_sim))

            logger.debug('after matmul cosine_sim shape : {}'.format(cosine_sim.shape))
            cosine_sim = tf.reshape(cosine_sim, [-1, 1])
            logger.debug('after matmul cosine_sim shape : {}'.format(cosine_sim.shape))
        return cosine_sim

    def customNorm(self, tensor, axis=1):
        print('customNorm : input tensor shape = ', tensor.shape)
        norm = tf.nn.l2_normalize(tensor, axis=axis)
        print('customNorm : after norm shape = ', norm.shape)

        return norm


if __name__ == "__main__":
    # f = FirstModel()
    # f.preprocessing()
    # f.makeModel()
    # f.saveModel()
    # f.predictTest()

    f = FirstModel()
    # f.loadModel()
    f.predictTestData()
    # f.predictAllTrainData()

    # Test().checkShape()
