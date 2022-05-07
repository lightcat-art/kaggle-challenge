import numpy as np
import pandas as pd
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
import datetime
from kaggle.common.config import Config
from kaggle.common.params import PARAM
import os

# 전역 공통 디렉토리 변수 선언
server_config = Config(None, PARAM.FILE_CONFIG)
PATH_HOME = server_config.get_param(PARAM.PARAM_PATH_HOME, "./")
RESOURCE_NAME = 'RS1'


class FirstModel:
    def __init__(self):
        self.train_1_X = None
        self.train_2_X = None
        self.train_y = None
        self.train_X_len = None
        self.anchor_len = None
        self.target_len = None
        self.vocab_size = None
        self.output_dim = 512
        self.model = None
        self.tokenizer = None

        pass

    def preprocessing(self):
        print('RESOURCE_NAME = ',RESOURCE_NAME)
        print('RS_HOME = ',server_config.get_param(PARAM.PARAM_RESOURCES_HOME))
        print('RS_FOLDER = ',server_config.get_param(PARAM.PARAM_RESOURCES_FOLDER))
        df = pd.read_csv(
            os.path.join(server_config.get_param(PARAM.PARAM_RESOURCES_HOME),
                         server_config.get_param(PARAM.PARAM_RESOURCES_FOLDER),
                         RESOURCE_NAME, "train.csv"))
        # print(df)
        # print(df['score'].unique())

        # print('data size = ', df.value_counts())
        print('data = ', df)
        self.train_1_X = df['anchor']
        self.train_2_X = df['target']
        self.train_y = df['score']
        result = pd.concat([self.train_1_X, self.train_2_X])
        print('result count = ', result.count())
        result.index = [i for i in range(0, result.count())]

        # print(result)

        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(result, target_vocab_size=10000)
        # print('tokenizer subword size = ', len(self.tokenizer.subwords))
        # self.vocab_size = len(self.tokenizer.subwords)
        print('tokenizer vocab size = ', self.tokenizer.vocab_size)
        self.vocab_size = self.tokenizer.vocab_size
        print(self.tokenizer.subwords[:100])

        train_X = [self.tokenizer.encode(elem) for elem in result]
        # print(type(train_X))
        # print(train_X)
        self.train_X_len = max([len(line) for line in train_X])
        train_X_max_val = max([max(line) for line in train_X])
        print('max_val on train_X : ', train_X_max_val)
        train_X = pad_sequences(train_X, maxlen=self.train_X_len)

        self.train_1_X = train_X[:len(df)]
        self.train_2_X = train_X[len(df):]

        # print('padding 전 anchor : ', self.tokenizer.decode(self.train_1_X[0]))
        # print('padding 전 target : ', self.tokenizer.decode(self.train_2_X[0]))

        # self.anchor_len = max([len(line) for line in self.train_1_X])
        # self.target_len = max([len(line) for line in self.train_2_X])
        # self.train_1_X = pad_sequences(self.train_1_X, maxlen=self.anchor_len)
        # self.train_2_X = pad_sequences(self.train_2_X, maxlen=self.target_len)

        print('padding 후 anchor shape : [', len(self.train_1_X), ',', len(self.train_1_X[0]), ']')
        print('padding 후 target shape : [', len(self.train_2_X), ',', len(self.train_2_X[0]), ']')
        print('label shape : ', self.train_y.shape)

    def makeModel(self):
        anchor = Input(shape=(self.train_X_len,))
        target = Input(shape=(self.train_X_len,))
        # Embedding층에 들어갈 입력 두개. 원핫인코딩이 되지 않은 상태이며,
        # 따라서 (batch_size, input_length) 인데 임베딩층을 거치면 (batch_size, input_length, embed_output_dim) 이 된다.

        print('model Input shape : ', anchor.shape, target.shape)

        anchor_embed = Embedding(self.vocab_size, self.output_dim, mask_zero=True
                                 , input_length=self.train_X_len
                                 )
        x = anchor_embed(anchor)
        print('anchor embed shape : ', x.shape)
        anchor_lstm = LSTM(units=256)  # return_state=True이면 list형태로 3개의 텐서, result, state_c, state_h 가 반환됨.
        anchor_output = anchor_lstm(x)

        target_embed = Embedding(self.vocab_size, self.output_dim, mask_zero=True
                                 , input_length=self.train_X_len
                                 )
        y = target_embed(target)
        print('target embed shape : ', y.shape)
        target_lstm = LSTM(units=256)
        target_output = target_lstm(y)

        print('lstm output shape : ', anchor_output.shape, target_output.shape)

        # x_norm = normalize(x, axis=2, order=1)
        # y_norm = normalize(y, axis=2, order=1)
        anchor_norm = self.customNorm(anchor_output, axis=1)
        anchor_norm = tf.expand_dims(anchor_norm, 1)
        # anchor_norm = tf.expand_dims(anchor_norm, 1)
        # anchor_norm = Flatten()(anchor_norm)
        print('anchor norm shape : ', anchor_norm.shape)

        target_norm = self.customNorm(target_output, axis=1)
        target_norm = tf.expand_dims(target_norm, 2)
        # target_norm = tf.expand_dims(target_norm, 1)
        # target_norm = Flatten()(target_norm)
        print('target norm shape : ', target_norm.shape)
        # print('transpose target shape : ',tf.transpose(target_norm).shape)
        # print(Flatten()(anchor_norm).shape)
        # print(Flatten()(target_norm).shape)

        # 그냥 * 로는 원하는 모양으로 행렬곱이 되지 않아 matmul 사용
        cosine_sim = tf.matmul(anchor_norm, target_norm)
        print('after matmul cosine_sim shape : ', cosine_sim.shape)
        cosine_sim = tf.reshape(cosine_sim, [-1, 1])
        print('after matmul cosine_sim shape : ', cosine_sim.shape)

        # 3d shape tensor 그 이상에서는 행렬곱을 하고싶은 위치만 transpose되어야 행렬곱 가능.
        # expand_dims로 shape 조정 (복잡할때는 transpose이용하여 shape 조정 가능)

        # cosine_sim = tf.matmul(anchor_norm, target_norm)

        # cosine_sim = Flatten()(cosine_sim)

        cosine_sim = (cosine_sim + 1) / 2

        self.model = Model(inputs=[anchor, target], outputs=cosine_sim)

        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mse', 'accuracy'])

        # try:
        self.model.fit(x=[self.train_1_X, self.train_2_X], y=self.train_y, batch_size=server_config.get_param(PARAM.PARAM_BATCH_SIZE),
                       epochs= server_config.get_param(PARAM.PARAM_EPOCHS)
                       , validation_split=0.2
                       )
        # except Exception as ex:
        #     print('error occured..', ex)

    def customNorm(self, tensor, axis=1):
        print('customNorm : input tensor shape = ', tensor.shape)
        # print(tensor)
        # reduceMean = tf.reduce_mean(tensor, axis=axis)
        norm = tf.norm(tensor, ord=1, axis=axis)
        print('customNorm : norm val shape = ', norm.shape)
        norm = tf.expand_dims(norm, 1)

        normTensor = tensor / norm

        print('customNorm : after norm = ', normTensor)
        print('customNorm : after norm shape = ', normTensor.shape)

        return tensor

    def saveModel(self):
        self.model.save("mseModel_", datetime.datetime.now(), ".h5")

    def load(self):
        self.model = keras.models.load_model("mseModel.h5")

        df = pd.read_csv('E:/Kaggle/U.S. Patent Phrase to Phrase Matching/train.csv')

        self.train_1_X = df['anchor']
        self.train_2_X = df['target']
        result = pd.concat([self.train_1_X, self.train_2_X])
        result.index = [i for i in range(0, result.count())]

        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(result, target_vocab_size=10000)

    def predictTest(self):
        anchor = self.train_1_X[0:2]
        target = self.train_2_X[0:2]
        anchor = self.tokenizer.decode(anchor)
        anchor = pad_sequences(anchor, maxlen=self.anchor_len)
        target = self.tokenizer.decode(target)
        target = pad_sequences(target, maxlen=self.target_len)
        sim = self.model.predict([anchor, target])
        print(sim)

    def predictModel(self, anchor, target):
        anchor = self.tokenizer.decode(anchor)
        anchor = pad_sequences(anchor, maxlen=self.anchor_len)
        target = self.tokenizer.decode(target)
        target = pad_sequences(target, maxlen=self.target_len)
        sim = self.model.predict([anchor, target])
        print(sim)


class Test:
    def __init__(self):
        self.output_dim = 512
        self.vocab_size = 12099
        pass

    def checkShape(self):
        # anchor = np.random.random((1111, 15))
        anchor = tf.random.uniform(shape=(1111, 15), minval=1, maxval=self.vocab_size, dtype=tf.dtypes.int32)
        # target = np.random.random((1111, 15))
        target = tf.random.uniform(shape=(1111, 15), minval=1, maxval=self.vocab_size, dtype=tf.dtypes.int32)
        # Embedding층에 들어갈 입력 두개. 원핫인코딩이 되지 않은 상태이며,
        # 따라서 (batch_size, input_length) 인데 임베딩층을 거치면 (batch_size, input_length, embed_output_dim) 이 된다.

        print('model Input shape : ', anchor.shape, target.shape)

        anchor_embed = Embedding(self.vocab_size, self.output_dim, mask_zero=True
                                 # , input_length=self.train_X_len
                                 )
        x = anchor_embed(anchor)
        print('anchor embed shape : ', x.shape)
        anchor_lstm = LSTM(units=256)  # return_state=True이면 list형태로 3개의 텐서, result, state_c, state_h 가 반환됨.
        anchor_output = anchor_lstm(x)

        target_embed = Embedding(self.vocab_size, self.output_dim, mask_zero=True
                                 # , input_length=self.train_X_len
                                 )
        y = target_embed(target)
        print('target embed shape : ', y.shape)
        target_lstm = LSTM(units=256)
        target_output = target_lstm(y)

        print('lstm output shape : ', anchor_output.shape, target_output.shape)

        # x_norm = normalize(x, axis=2, order=1)
        # y_norm = normalize(y, axis=2, order=1)
        anchor_norm = self.customNorm(anchor_output, axis=1)
        anchor_norm = tf.expand_dims(anchor_norm, 1)
        # anchor_norm = tf.expand_dims(anchor_norm, 1)
        # anchor_norm = Flatten()(anchor_norm)
        print('anchor norm shape : ', anchor_norm.shape)

        target_norm = self.customNorm(target_output, axis=1)
        target_norm = tf.expand_dims(target_norm, 2)
        # target_norm = tf.expand_dims(target_norm, 1)
        # target_norm = Flatten()(target_norm)
        print('target norm shape : ', target_norm.shape)
        # print('transpose target shape : ',tf.transpose(target_norm).shape)
        # print(Flatten()(anchor_norm).shape)
        # print(Flatten()(target_norm).shape)

        # 그냥 * 로는 원하는 모양으로 행렬곱이 되지 않아 matmul 사용
        cosine_sim = tf.matmul(anchor_norm, target_norm)
        print('after matmul cosine_sim shape : ', cosine_sim.shape)
        print('after matmul cosine_sim : ', cosine_sim)
        cosine_sim = tf.reshape(cosine_sim, [-1, 1])
        print('after matmul cosine_sim shape : ', cosine_sim.shape)
        print('after matmul cosine_sim : ', cosine_sim)

        # 3d shape tensor 그 이상에서는 행렬곱을 하고싶은 위치만 transpose되어야 행렬곱 가능.
        # expand_dims로 shape 조정 (복잡할때는 transpose이용하여 shape 조정 가능)

        # cosine_sim = tf.matmul(anchor_norm, target_norm)

        # cosine_sim = Flatten()(cosine_sim)

        print(cosine_sim)
        print('min value : ', min(cosine_sim), ', max value : ', max(cosine_sim))
        cosine_sim = (cosine_sim + 1) / 2
        print(cosine_sim)
        print('min value : ', min(cosine_sim), ', max value : ', max(cosine_sim))

        print('cosine sim shape : ', cosine_sim.shape)

    def customNorm(self, tensor, axis=1):
        print('customNorm : input tensor shape = ', tensor.shape)
        # print(tensor)
        # reduceMean = tf.reduce_mean(tensor, axis=axis)
        norm = tf.norm(tensor, ord=1, axis=axis)
        print('customNorm : norm val shape = ', norm)
        norm = tf.expand_dims(norm, 1)

        normTensor = tensor / norm

        print('customNorm : after norm = ', normTensor)
        print('customNorm : after norm shape = ', normTensor.shape)

        return tensor


if __name__ == "__main__":
    f = FirstModel()
    f.preprocessing()
    f.makeModel()
    f.saveModel()
    f.predictTest()

    # Test().checkShape()
    # print(ResourceManager().getResourcePath())
