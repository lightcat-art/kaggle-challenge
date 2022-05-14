import re
import os
import json

"""
    modelConfig.json 파일 작성방법
"""
"""
{
    "[TASK_NAME_1]": {
        "[MODEL_TYPE_1]": {
            "MODEL_NAME": "MseModel",
            "BATCH_SIZE": 64,
            "EPOCHS": 1,
            "OPTIMIZER": "rmsprop",
            "LOSS": "mse",
            "METRICS": "accuracy",
            "VAL_SPLIT": 0.2
            },
        "[MODEL_TYPE_2]": {
            "MODEL_NAME": "MseModel2",
            "BATCH_SIZE": 64,
            "EPOCHS": 200,
            "OPTIMIZER": "rmsprop",
            "LOSS": "mse",
            "METRICS": "accuracy",
            "VAL_SPLIT": 0.2
            }
    },
    "[TASK_NAME_1]": {
        "[MODEL_TYPE_1]": {
            "MODEL_NAME": "MseModel",
            "BATCH_SIZE": 64,
            "EPOCHS": 1,
            "OPTIMIZER": "rmsprop",
            "LOSS": "mse",
            "METRICS": "accuracy",
            "VAL_SPLIT": 0.2
            },
        "[MODEL_TYPE_2]": {
            "MODEL_NAME": "MseModel2",
            "BATCH_SIZE": 64,
            "EPOCHS": 200,
            "OPTIMIZER": "rmsprop",
            "LOSS": "mse",
            "METRICS": "accuracy",
            "VAL_SPLIT": 0.2
            }
    }
}
"""
class ModelConfigManager:
    """
    모델 학습 관리자
    """
    __instance = None
    log = '[LearningManager] '

    def __new__(cls, *args, **kwargs):
        """
        싱글톤 동작
        """
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        return cls.__instance

    def init(self, default=None, json_file=None):
        """
        인자로 dict나 json 형식의 파일이름을 받아 데이터를 설정한다.
        먼저 dict 형식의 인자를 추가하고, 그 다음에 json 형식의 파일을 읽어
        내용을 추가하거나 갱신한다.

        Args:
            default (dict or None) : json type dict data.
            json_file (str or None) : json file path.
        """

        # default값 먼저 설정
        if default is not None and type(default) == dict:
            self.update_from_dict(default)

        if json_file is not None and type(json_file) == str:
            self.load_json(json_file)

    def _clean_name(self, name):
        """
        받은 문자열을 변수로 사용할 수 있는 문자열로 바꾼다.

        Args:
            name (str) : 변수로 사용할 문자열.

        Returns:
            str : 변경된 문자열. 실패한 경우에는 ""가 반환된다.
        """
        result = name
        # 앞에 붙은 '-'를 제거한다.
        while result.startswith("-"):
            result = result[1:]

        # '_', '.', ":" 를 제외한 특수문자가 남아있는지 확인한다.
        if len(re.findall("[^a-zA-Z0-9_.:/\\\]+", result)) > 0:
            return ""

        return result

    def update_from_dict(self, dict_data):
        """
        dict에 있는 데이터를 변수로 추가한다.
        기존에 있는 데이터는 새로운 값으로 갱신한다.

        Args:
            dict_data (dict) : json 형식의 dict 데이터
        Returns:
            boolean : 성공하면 True를 실패하면 False를 반환한다.
        """
        if type(dict_data) != dict:
            return False

        for key in dict_data:
            # 값의 이름을 쓸수 있는 형태로 바꾼다.
            name = self._clean_name(str(key.strip()))
            if name == "":
                continue

            # 값을 설정한다
            value = dict_data[key]
            setattr(self, name, value)

        return True



    def load_json(self, filename):
        """
        json 파일을 읽어 설정을 갱신한다.

        Args:
            filename (str) : json filename.

        Returns:
            bool : True or False
        """
        if not (type(filename) == str and filename != "" and os.path.isfile(filename)):
            print(self.log, 'filename is abnormal or path is abnoraml')
            return False
        try:
            with open(filename, "r", encoding="utf=8") as rf:
                data = json.load(rf)

        except Exception as e:
            print(self.log, 'load_json error occured..', e)
            return False

        self.update_from_dict(data)

        return True

    def get_task_model_config(self, task_name, model_type):
        """
        특정모델이 사용하는 config dict를 반환
        :param task_name:
        :param model_type:
        :return: dict 형태의 config
        """
        try:
            if not hasattr(self, task_name):
                return None
        except Exception as e:
            print(self.log, 'get_model_config error occured..', e)
            return None

        return getattr(self, task_name).get(model_type)

    def get_task_config(self, task_name):
        """
        특정모델이 사용하는 config dict를 반환
        :param task_name:
        :return: dict 형태의 config
        """
        try:
            if not hasattr(self, task_name):
                return None
        except Exception as e:
            print(self.log, 'get_package_model_config error occured..', e)
            return None

        return getattr(self, task_name)

    def get_all_params(self):
        """
        가진 모든 변수를 dict형태로 반환한다.
        가져간 dict 데이터를 수정해도 Config에는 반영되지 않는다.

        Returns:
            dict : 가진 모든 변수의 dict 데이터
        """
        return self.__dict__.copy()
