from typing import List

from kaggle.common.logManager import LogManager
from kaggle.patentmatching import build
from kaggle.common.modelConfigManager import ModelConfigManager
from kaggle.common.params import PARAM
from kaggle.common.classificationCode import TaskMap
import os
import sys

# 전역 공통 디렉토리 변수 선언
model_config = ModelConfigManager()
model_config.init(None, PARAM.FILE_MODEL_CONFIG)

# 모델 학습
OPTION_LEARNING = "--learn"

# 모델 예측
OPTION_PREDICT = "--predict"

# 테스트
OPTION_TEST = "--test"

# GRPC 예측
OPTION_GRPC = "--grpc"

LOGGER_NAME = "MAIN"
logger = LogManager().get_logger(LOGGER_NAME, PARAM.LOG_COMMON_FILE)


# server_config = Config(None, PARAM.FILE_CONFIG)
# print('get all param : ',server_config.get_all_params())
# PATH_HOME = server_config.get_param(PARAM.PARAM_PATH_HOME, "./")
# print('PATH_HOME=',PATH_HOME)

def validate(task_name: str, model_type: str):
    if not model_config.get_task_config(task_name) is None:
        if not model_config.get_task_model_config(task_name, model_type) is None:
            return True
        logger.error('model_type isn''t exist')
    else:
        logger.error('task_name isn''t exist')
    return False


def runTask(TASK_NAME, MODEL_TYPE, option):
    if TASK_NAME == TaskMap.PATENTMATCHING.name:
        if MODEL_TYPE == TaskMap.PATENTMATCHING.value.get('MSE'):
            if option == OPTION_LEARNING:
                f = build.FirstModel()
                f.preprocessing()
                f.makeModel()
                f.saveModel()
            elif option == OPTION_PREDICT:
                f = build.FirstModel()
                f.preprocessing()
                f.loadModel()
                f.predictTestData()
            elif option == OPTION_TEST:
                f = build.Test()
                f.checkShape()
            elif option == OPTION_GRPC:
                f = build.FirstModel()
                f.preprocessing()
                f.predictByGrpc()


if __name__ == "__main__":
    args = os.sys.argv

    # start.py 인자까지 argument로 인식함
    if not len(args) == 4:
        sys.exit('not enough argument length.')

    task_name = args[1]
    model_type = args[2]
    option = args[3]

    if validate(task_name, model_type):
        runTask(task_name, model_type, option)
    else:
        sys.exit('config validate fail. doesn''t exist TASK_NAME or MODEL_TYPE. Check ''modelConfig.json''')
