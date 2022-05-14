"""
로그를 처리하는 모듈

로거 오브젝트를 생성, 취득, 삭제등의
관리를 담당한다.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from absl import logging as absl_logging

from kaggle.common.pathManager import PathManager
from kaggle.common.config import Config
from kaggle.common.params import PARAM
from kaggle.common.languagePack import LANGPACK


class LogManager:
    """
    로그 파일의 생성과 관리를 돕는 클래스.
    이 클래스는 클래스 메소드만으로 구성되어 있어 싱글톤처럼 동작한다.
    """
    __instance = None
    __LOGGERS = None
    __HANDLERS = None
    _log_path = None
    _log_level = None
    _log_level_map = {
        "CRITICAL": logging.CRITICAL,
        "FATAL": logging.FATAL,
        "ERROR": logging.ERROR,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG
    }

    def __new__(cls, *args, **kwargs):
        """
        싱글톤 동작
        """
        if cls.__instance is None:
            cls.__instance = object.__new__(cls, *args, **kwargs)
        if cls.__LOGGERS is None:
            cls.__LOGGERS = {}
        if cls.__HANDLERS is None:
            cls.__HANDLERS = {}

        return cls.__instance

    @classmethod
    def get_logger(cls, logger_name, filename=None):
        """
        Pool에 있는 로거를 반환한다. 만약 그 로거가 처음 call 되는 것이면,
        로거를 생성하고 Pool에 추가한 뒤에 반환한다.

        Args:
            logger_name (str) : 로거 이름. 이 이름을 기준으로 파일을 생성한다.
            filename (str) : 로그를 저장할 파일명. 없으면 로거 이름을 가져다 쓴다.

        Returns:
            logger : 로거 오브젝트.
        """
        if cls._log_path is None:
            cls._log_path = PathManager().get_log_path()

        if cls._log_level is None:
            log_level = Config(None, PARAM.FILE_CONFIG).get_param(PARAM.PARAM_LOG_LEVEL, "DEBUG")
            cls._log_level = cls._log_level_map.get(log_level)

        # 인자 확인
        if logger_name is None or logger_name == "":
            return None

        # 이미 있는 것이면 Pool에서 바로 반환.
        if logger_name in cls.__LOGGERS:
            return cls.__LOGGERS[logger_name]

        if filename is None:
            filename = logger_name + ".log"

        else:
            # cls._log_path는 디렉토리이고, filename에 디렉토리설정이 있는지 판단하여 isdir로 디렉토리존재여부 확인하고 없으면 makedir
            file_path = os.path.join(cls._log_path, os.path.dirname(filename))
            if not os.path.isdir(file_path):
                os.makedirs(file_path, exist_ok=True)

        if filename not in cls.__HANDLERS:
            file_handler = RotatingFileHandler(os.path.join(cls._log_path, filename),
                                               mode="wt",
                                               maxBytes=PARAM.LOG_MAX_BYTES,
                                               backupCount=PARAM.LOG_BACKUP_COUNT,
                                               encoding="utf-8")
            file_handler.setFormatter(logging.Formatter(PARAM.LOG_FORMAT, style="{"))
            file_handler.setLevel(cls._log_level)
            cls.__HANDLERS[filename] = file_handler
        else:
            file_handler = cls.__HANDLERS[filename]

        # 없으면 새로 생성
        logger = logging.getLogger(logger_name)
        logger.addHandler(file_handler)
        logger.setLevel(cls._log_level)
        logger.info(LANGPACK.START_LOG.format(logger_name, filename))

        cls.__LOGGERS[logger_name] = logger

        return logger


    @classmethod
    def close_logger(cls, logger_name):
        """
        해당하는 로거를 닫고 Pool에서 제거한다.

        Args:
            logger_name (str) : 삭제할 로거의 이름.

        Returns:
            boolean : 성공여부.
        """
        # 인자를 확인한다.
        if logger_name is None or logger_name == "":
            return False

        # 로거가 Pool에 있는지 확인
        if logger_name not in cls.__LOGGERS:
            return True

        #로거 닫기
        logger = cls.__LOGGERS.pop(logger_name)

        logger.info(LANGPACK.END_LOG.format(logger_name))

        logger.flush()
        logger.close()

        del logger

        return True
