import logging

from decorators.singleton import Singleton
from utils.utils_time import TimeUtils
from utils.utils_pid import PidUtils
from utils.constants import *


@Singleton
class LogUtils:

    def __init__(self):

        self._extra = {'pid': PidUtils.instance().get_pid()}
        self._set_logger = False

    def set_log_path(self, log_path: str):

        formatter = logging.Formatter(
            '%(asctime)s %(pid)s %(levelname)s %(message)s')
        start_time = TimeUtils.instance().get_start_time()

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)

        made_handler = logging.FileHandler(
            "{}/{}_log_made.log".format(log_path, start_time))
        made_handler.setFormatter(formatter)

        made_logger = logging.getLogger(LOGGER_MADE)
        made_logger.addHandler(made_handler)
        made_logger.addHandler(stream_handler)

        default_handler = logging.FileHandler(
            "{}/{}_log_general.log".format(log_path, start_time))
        default_handler.setFormatter(formatter)

        default_logger = logging.getLogger(LOGGER_DEFAULT)
        default_logger.addHandler(default_handler)
        default_logger.addHandler(stream_handler)

        self._set_logger = True

    def log_info(self, msg, type=LOGGER_DEFAULT):
        assert self._set_logger == True
        logger = logging.getLogger(type)
        logger = logging.LoggerAdapter(logger, self._extra)
        logger.setLevel(logging.INFO)
        logger.info(msg)

    def log_warning(self, msg, type=LOGGER_DEFAULT):
        assert self._set_logger == True
        logger = logging.getLogger(type)
        logger = logging.LoggerAdapter(logger, self._extra)
        logger.setLevel(logging.WARN)
        logger.warn(msg)

    def log_error(self, msg, type=LOGGER_DEFAULT):
        assert self._set_logger == True
        logger = logging.getLogger(type)
        logger = logging.LoggerAdapter(logger, self._extra)
        logger.setLevel(logging.ERROR)
        logger.error(msg)
