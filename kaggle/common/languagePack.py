import os
import json
from kaggle.common.params import PARAM


class LangPack:
    """
    딥러닝에서 사용하는 문자열의 처리를 한다.
    문자열은 config/language.json 에 저장된 것을 사용한다.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)
            cls._instance._load()
        return cls._instance

    def _load(self):
        """
        config/language.json 파일을 읽어 내부 변수들로 설정한다.
        """
        langpack_filename = PARAM.FILE_LANGPACK_CONFIG
        print('LANGPACK FILENAME=',langpack_filename)
        if not os.path.isfile(langpack_filename):
            raise RuntimeError("Can't find config/language.json")

        with open(langpack_filename, "r", encoding="utf-8") as rf:
            lang_dict = json.load(rf)

        for name, value in lang_dict.items():
            setattr(self, name, value)


LANGPACK = LangPack()
