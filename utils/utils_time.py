import time
from decorators.singleton import Singleton


@Singleton
class TimeUtils:

    def __init__(self):
        self._start_time = time.strftime("%Y%m%d-%H%M%S")

    def get_start_time(self):
        return self._start_time
