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
    def PACKAGE_NAME(self):
        packagePath = os.getcwd().split('\\')
        print(packagePath)
        return packagePath[len(packagePath) - 1]

    # 기본 파일명
    @property
    def FILE_CONFIG(self):
        return os.path.join(".", Params().PATH_CONFIG, "config.json")

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

    @property
    def PARAM_EPOCHS(self):
        return "EPOCHS"

    @property
    def PARAM_BATCH_SIZE(self):
        return "BATCH_SIZE"


# 여기서 그냥 instance 생성해둠.
PARAM = Params()

if __name__ == "__main__":
    print(PARAM.PARAM_RESOURCES_HOME)
