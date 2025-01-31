from abc import ABC, abstractmethod

from model.result import Result


class AbstractClassifierService(ABC):

    @abstractmethod
    def run_classification(self, **kwargs) -> Result:
        raise NotImplementedError("Abstract method shall not be invoked!")

    @abstractmethod
    def is_eligible(self, type: str) -> bool:
        raise NotImplementedError("Abstract method shall not be invoked!")
