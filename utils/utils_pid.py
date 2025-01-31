import uuid
from decorators.singleton import Singleton


@Singleton
class PidUtils:

    def __init__(self):
        self._pid = uuid.uuid4()

    def get_pid(self):
        return self._pid
