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

server_config = Config(None, PARAM.FILE_CONFIG)
path_manager = PathManager()
model_config = ModelConfigManager()
PATH_HOME = server_config.get_param(PARAM.PARAM_PATH_HOME, "./")
TASK_NAME = TaskMap.PATENTMATCHING.name  # modelConfig.json의 TASK_NAME과 동일해야함
logger = LogManager().get_logger(TASK_NAME, PARAM.LOG_COMMON_FILE)


class KafkaInference:
    def __init__(self):
        self.tokenizer = None
        self.preprocess_info = {}
        self.MODEL_TYPE = TaskMap.PATENTMATCHING.value.get('MSE')  # modelConfig.json의 MODEL_TYPE과 동일해야함

    def loadTokenizerDic(self):
        self.tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(
            os.path.join(path_manager.get_data_path(TASK_NAME, self.MODEL_TYPE), PARAM.PARAM_TOKENIZER_FILE_NAME))

    def loadPreprocessInfo(self):
        prepro_info_file = os.path.join(path_manager.get_data_path(TASK_NAME, self.MODEL_TYPE),
                                        PARAM.PARAM_PREPROCESS_INFO_FILE_NAME)
        with open(prepro_info_file, 'r', encoding='utf-8') as rf:
            data = json.load(rf)

        self.preprocess_info.update(data)

    def predictByGrpc(self, anchor, target):
        localhost = 'localhost'
        port = '8500'

        # 들어온 데이터에 대한 토크나이저 인코딩 및 패딩 필요!!!
        # anchor = self.train_1_X[0:2]
        # target = self.train_2_X[0:2]

        logger.debug('anchor shape : ({},{})'.format(len(anchor), len(anchor[0])))
        logger.debug('target shape : ({},{})'.format(len(target), len(target[0])))
        simScore = self.train_y[0:2]
        logger.debug('simScore : {}'.format(simScore))

        with grpc.insecure_channel(f'{localhost}:{port}') as channel:
            stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

            get_model_metadata_request = get_model_metadata_pb2.GetModelMetadataRequest()
            # 모델명 설정 필요
            get_model_metadata_request.model_spec.name = 'MSE'
            get_model_metadata_request.metadata_field.append('signature_def')
            resp = stub.GetModelMetadata(get_model_metadata_request, 5.0)  # 5 secs timeout

            signature_def = resp.metadata['signature_def']
            signature_map = get_model_metadata_pb2.SignatureDefMap()
            print("PredictGrpc : signature_def value = {}".format(signature_def.value))
            signature_map.ParseFromString(signature_def.value)

            serving_default = signature_map.ListFields()[0][1]['serving_default']
            serving_inputs = serving_default.inputs
            serving_outputs = serving_default.outputs
            print("PredictGrpc : serving_inputs = {}".format(serving_inputs))
            print("PredictGrpc : serving_outputs = {}".format(serving_outputs))

            # input = self.get_input_output(datas=serving_inputs)
            # print("get_input_output : input keys = {}".format(key))
            # output = self.get_input_output(datas=serving_outputs)
            # print("get_input_output : output keys = {}".format(key))

            request = predict_pb2.PredictRequest()
            # 모델명 설정 필요
            request.model_spec.name = 'MSE'
            request.model_spec.signature_name = 'serving_default'
            # CopyFrom의 인자는 필수적으로 TensorProto 형태가 되어야함.
            # 아니면 TypeError: Parameter to CopyFrom() must be instance of same class: expected tensorflow.TensorProto got list. 에러발생
            # 각 input이 Tensor형태여도 안받아들여짐.
            # request.inputs[input].CopyFrom(tf.make_tensor_proto({'input_1':anchor,'input_2':target}))
            for key in serving_inputs.keys():
                print("PredictGrpc : serving_input element = {}".format(key))
                if 'anchor' in key:
                    request.inputs[key].CopyFrom(tf.make_tensor_proto(anchor, dtype=tf.dtypes.float32))
                elif 'target' in key:
                    request.inputs[key].CopyFrom(tf.make_tensor_proto(target, dtype=tf.dtypes.float32))
            result = stub.Predict(request, 120.0)  # 120 secs timeout

            response = {}
            for key in serving_outputs.keys():
                print("PredictGrpc : serving_output element = {}".format(key))
                response.update({key: np.array(result.outputs[key].float_val)})

            print(response)
