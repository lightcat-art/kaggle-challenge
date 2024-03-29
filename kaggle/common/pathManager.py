from kaggle.common.params import PARAM
from kaggle.common.config import Config
from kaggle.common.modelConfigManager import ModelConfigManager
import os
import traceback



class PathManager:
    """
    딥러닝에서 사용하는 모든 Path를 관리한다.
    """

    __instance = None

    _PATH_HOME = "./"
    _PATH_LOG = "logs"
    _PATH_MODEL = "model"
    _PATH_RESOURCES_HOME = "."
    _PATH_RESOURCES_FOLDER = "resources"

    def __new__(cls, *args, **kwargs):
        """
        Singleton 구성
        """
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
            cls.__instance.init()

        return cls.__instance

    def init(self):
        """
        설정 파일에서 변경 가능한 경로를 읽어들인다.
        """
        server_config = Config(None, PARAM.FILE_CONFIG)
        self._PATH_HOME = server_config.get_param(PARAM.PARAM_PATH_HOME, self._PATH_HOME)
        self._PATH_LOG = server_config.get_param(PARAM.PARAM_PATH_LOGS, self._PATH_LOG)
        self._PATH_RESOURCES_HOME = server_config.get_param(PARAM.PARAM_RESOURCES_HOME, self._PATH_RESOURCES_HOME)
        self._PATH_RESOURCES_FOLDER = server_config.get_param(PARAM.PARAM_RESOURCES_FOLDER, self._PATH_RESOURCES_FOLDER)

    def _check_path(self, target_path):
        """
        대상 경로가 있는지 확인하고 없으면 해당 폴더를 생성한다.

        Args:
            target_path (str) : 확인할 대상 경로.
        """

        # 해당 폴더가 없으면 생성
        if not os.path.isdir(target_path):
            try:
                os.makedirs(target_path, exist_ok=True)
            except Exception as e:
                traceback.print_exc()

    def _clear_file(self, target_path):
        """
        폴더경로가 있는지 확인하고 있으면 폴더 내 파일을 지움
        :param target_path:
        :return:
        """
        if os.path.isdir(target_path):
            try:
                for file in os.scandir(target_path):
                    os.remove(file.path)
                return True
            except Exception as e:
                traceback.print_exc()
        else:
            return False

    def get_log_path(self):
        """
        로그 폴더를 반환한다.

        Returns:
            str : 로그 폴더의 전체 경로
        """
        target_path = os.path.abspath(os.path.join(self._PATH_HOME, self._PATH_LOG))
        self._check_path(target_path)

        return target_path

    # def get_resources_path(self):
    #     target_path = os.path.abspath(os.path.join(self._PATH_RESOURCES_HOME, self._PATH_RESOURCES_FOLDER))
    #     self._check_path(target_path)
    #
    #     return target_path

    def get_task_path(self, task_name):
        target_path = os.path.abspath(os.path.join(self._PATH_RESOURCES_HOME, self._PATH_RESOURCES_FOLDER, task_name))
        self._check_path(target_path)

        return target_path

    def get_task_model_type_path(self, task_name, model_type):
        target_path = os.path.abspath(
            os.path.join(self._PATH_RESOURCES_HOME, self._PATH_RESOURCES_FOLDER, task_name, model_type))
        self._check_path(target_path)

        return target_path

    def get_data_path(self, task_name, model_type):
        """
        학습시 사용하는 모델별 데이터 경로
        :param task_name:
        :param model_type:
        :return:
        """
        target_path = os.path.abspath(
            os.path.join(self._PATH_RESOURCES_HOME, self._PATH_RESOURCES_FOLDER, task_name, model_type,
                         PARAM.PATH_DATA))
        self._check_path(target_path)

        return target_path

    def get_model_path(self, task_name, model_type):
        """
        TASK에서 사용하는 모델폴더 경로

        Returns:
            str : 대상 모델 폴더의 전체 경로
        """
        target_path = os.path.abspath(
            os.path.join(self._PATH_RESOURCES_HOME, self._PATH_RESOURCES_FOLDER, task_name, model_type,
                         PARAM.PATH_MODEL))
        self._check_path(target_path)

        return target_path

    def get_model_data_path(self, task_name, model_type, model_name):
        """
        지정된 모델이름으로 된 폴더 경로

        Returns:
            str : 대상 모델 폴더의 전체 경로
        """
        target_path = os.path.abspath(
            os.path.join(self._PATH_RESOURCES_HOME, self._PATH_RESOURCES_FOLDER, task_name, model_type,
                         PARAM.PATH_MODEL, model_name))
        self._check_path(target_path)

        return target_path

    def get_checkpoint_path(self, task_name, model_type, model_name, clear_option=False):
        """
        TASK의 체크포인트 저장할 폴더 경로

        Returns:
            str :
        """
        target_path = os.path.abspath(
            os.path.join(self._PATH_RESOURCES_HOME, self._PATH_RESOURCES_FOLDER, task_name, model_type,
                         PARAM.PATH_CHECKPOINT, model_name))
        self._check_path(target_path)
        if clear_option:
            clearResult = self._clear_file(target_path)
            print('clear result = {}'.format(clearResult))

        return target_path
