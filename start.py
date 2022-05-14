from typing import List

from kaggle.patentmatching import build
import os

# 전역 공통 디렉토리 변수 선언
# server_config = Config(None, PARAM.FILE_CONFIG)
# print('get all param : ',server_config.get_all_params())
# PATH_HOME = server_config.get_param(PARAM.PARAM_PATH_HOME, "./")
# print('PATH_HOME=',PATH_HOME)

if __name__ == "__main__":
    f = build.FirstModel()
    f.preprocessing()
    f.makeModel()
    f.saveModel()

    # f = build.FirstModel()
    # f.loadModel()
    # # f.predictTest()
    # f.predictAllTrainData()