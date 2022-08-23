from kaggle.common.classificationCode import TaskMap
from kaggle.common.config import Config
from kaggle.common.logManager import LogManager
from kaggle.common.modelConfigManager import ModelConfigManager
from kaggle.common.params import PARAM
from kaggle.common.pathManager import PathManager
import numpy as np
import tensorflow as tf
import grpc
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import get_model_metadata_pb2
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import predict_pb2
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds
import json
# kafka
from kafka import KafkaProducer, KafkaConsumer

server_config = Config(None, PARAM.FILE_CONFIG)
path_manager = PathManager()
model_config = ModelConfigManager()
PATH_HOME = server_config.get_param(PARAM.PARAM_PATH_HOME, "./")
TASK_NAME = TaskMap.PATENTMATCHING.name  # modelConfig.json의 TASK_NAME과 동일해야함
logger = LogManager().get_logger(TASK_NAME, PARAM.LOG_COMMON_FILE)


class Producer:
    def __init__(self):
        self.tokenizer = None
        self.preprocess_info = {}
        self.MODEL_TYPE = TaskMap.PATENTMATCHING.value.get('MSE')  # modelConfig.json의 MODEL_TYPE과 동일해야함
        self.TOPIC_NAME = 'kaggle-patent-matching'
        # serialize our data to json for efficient transfer
        self.producer = KafkaProducer(value_serializer=lambda msg: json.dumps(msg).encode('utf-8'),
                                      bootstrap_servers=['localhost:9092'])
        self.loadTokenizerDic()
        self.loadPreprocessInfo()

    def loadTokenizerDic(self):
        logger.debug('load tokenizer dic.')
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            os.path.join(path_manager.get_data_path(TASK_NAME, self.MODEL_TYPE), PARAM.PARAM_TOKEN_FILE_NAME))

    def loadPreprocessInfo(self):
        logger.debug('load preprocess info.')
        prepro_info_file = os.path.join(path_manager.get_data_path(TASK_NAME, self.MODEL_TYPE),
                                        PARAM.PARAM_PREPROCESS_INFO_FILE_NAME)
        with open(prepro_info_file, 'r', encoding='utf-8') as rf:
            data = json.load(rf)

        self.preprocess_info.update(data)

    def preprocessing(self, anchor, target):
        anchor = [self.tokenizer.encode(elem) for elem in anchor]
        target = [self.tokenizer.encode(elem) for elem in target]
        anchor = pad_sequences(anchor, maxlen=self.preprocess_info.get(PARAM.PARAM_MAX_PAD_LEN))
        target = pad_sequences(target, maxlen=self.preprocess_info.get(PARAM.PARAM_MAX_PAD_LEN))
        return anchor, target

    def produceEvent(self):
        """
        Function to produce events
        """
        # UUID produces a universal unique identifier
        anchor = ['good choice', 'what a good man']
        target = ['good select', 'what a nice boy']
        anchor, target = self.preprocessing(anchor, target)
        return {
            'anchor': anchor.tolist(),
            'target': target.tolist()
        }

    def sendEvents(self):
        data = self.produceEvent()
        print('sending = {}'.format(data))
        result = self.producer.send(self.TOPIC_NAME, value=data)


class Consumer:
    def __init__(self):
        self.TOPIC_NAME = 'kaggle-patent-matching-output'
        self.consumer = KafkaConsumer(self.TOPIC_NAME,
                                      auto_offset_reset='earliest',  # where to start reading the messages at
                                      group_id='kaggle-output', bootstrap_servers=['localhost:9092'],
                                      value_deserializer=lambda m: json.loads(m.decode('utf-8')))

    def consumeEvents(self):
        for m in self.consumer:
            print("received : {}".format(m.value))
