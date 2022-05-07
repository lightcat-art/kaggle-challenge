from typing import List

from kaggle.common.config import Config
from kaggle.patentmatching import build
import os
from kaggle.common.params import PARAM

# 전역 공통 디렉토리 변수 선언
# server_config = Config(None, PARAM.FILE_CONFIG)
# print('get all param : ',server_config.get_all_params())
# PATH_HOME = server_config.get_param(PARAM.PARAM_PATH_HOME, "./")
# print('PATH_HOME=',PATH_HOME)

def _initialize(args: List[str]):
    """
    설정 초기화작업 진행
    :param args: 프로그램 실행 파라미터
    :return:
    """

    config_path = os.path.dirname(PARAM.FILE_CONFIG)
    if not os.path.isdir(config_path):
        os.makedirs(config_path, exist_ok=True)

    default_server_config = Config()
    default_server_config.save_json(PARAM.FILE_CONFIG)


if __name__ == "__main__":
    f = build.FirstModel()
    f.preprocessing()
    f.makeModel()
    f.saveModel()
    # f.predictTest()
