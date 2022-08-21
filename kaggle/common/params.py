"""
파라미터 관련 설정 값.
"""
import os


class Params:
    """
    프로그램 전역에서 사용하는 파라미터 이름 관리

    setter를 지원하지 않는 함수가 선언되어 있음
    """

    __instance = None

    def __new__(cls, *args, **kwargs):
        """
        싱글톤 동작을 위해 오버라이딩
        """
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)

        return cls.__instance

    # 기본 경로값
    @property
    def PATH_CONFIG(self):
        return "config"

    @property
    def PATH_MODEL(self):
        return "model"

    @property
    def PATH_CHECKPOINT(self):
        return "checkpoint"

    @property
    def PATH_DATA(self):
        return "data"

    @property
    def CUR_PACKAGE_NAME(self):
        packagePath = os.getcwd().split('\\')
        # print(packagePath)
        return packagePath[len(packagePath) - 1]

    # 로그 관련 설정값
    @property
    def LOG_FORMAT(self):
        return "{asctime} [{levelname[0]}] > {message}    < in [{filename}:{lineno:04}]"

    @property
    def LOG_MAX_BYTES(self):
        return 1024 * 1024 * 100  # 100MB

    @property
    def LOG_BACKUP_COUNT(self):
        return 10

    @property
    def LOG_COMMON_FILE(self):
        return "server.log"

    # 기본 파일명
    @property
    def FILE_CONFIG(self):
        return os.path.join(".", Params().PATH_CONFIG, "serverConfig.json")

    @property
    def FILE_LANGPACK_CONFIG(self):
        return os.path.join(".", Params().PATH_CONFIG, "language.json")

    @property
    def FILE_MODEL_CONFIG(self):
        return os.path.join(".", Params().PATH_CONFIG, "modelConfig.json")

    # 파라미터 이름
    @property
    def PARAM_PATH_HOME(self):
        return "PATH_HOME"

    @property
    def PARAM_PATH_LOGS(self):
        return "PATH_LOGS"

    @property
    def PARAM_LOG_LEVEL(self):
        return "LOG_LEVEL"

    @property
    def PARAM_RESOURCES_HOME(self):
        return "RESOURCES_HOME"

    @property
    def PARAM_RESOURCES_FOLDER(self):
        return "RESOURCES_FOLDER"

    # 학습관련 파라미터
    @property
    def PARAM_PACKAGE_NAME(self):
        return "PACKAGE_NAME"

    @property
    def PARAM_LEARN_TYPE(self):
        return "LEARN_TYPE"

    @property
    def PARAM_MODEL_NAME(self):
        return "MODEL_NAME"

    @property
    def PARAM_MODEL_CHECKPOINT_NAME(self):
        return "MODEL_CHECKPOINT_NAME"

    @property
    def PARAM_BATCH_SIZE(self):
        return "BATCH_SIZE"

    @property
    def PARAM_EPOCHS(self):
        return "EPOCHS"

    @property
    def PARAM_OPTIMIZER(self):
        return "OPTIMIZER"

    @property
    def PARAM_LOSS(self):
        return "LOSS"

    @property
    def PARAM_METRICS(self):
        return "METRICS"

    @property
    def PARAM_VAL_SPLIT(self):
        return "VAL_SPLIT"

    # PREPROCESS_INFO 관련 파라미터
    @property
    def PARAM_MAX_PAD_LEN(self):
        return "MAX_PAD_LEN"

    @property
    def PARAM_TOKENIZER_FILE_NAME(self):
        return "tokenizer_save.dat"

    @property
    def PARAM_PREPROCESS_INFO_FILE_NAME(self):
        return "preprocess_info.dat"


# 여기서 그냥 instance 생성해둠.
PARAM = Params()

if __name__ == "__main__":
    print(PARAM.PARAM_RESOURCES_HOME)
    print(PARAM.PACKAGE_NAME)
