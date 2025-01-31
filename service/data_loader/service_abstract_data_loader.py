from abc import ABC, abstractmethod
from typing import Dict, Tuple
import numpy as np


class AbstractDataLoaderService(ABC):

    @abstractmethod
    def load_data(self, **kwargs) -> Dict[str, Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        raise NotImplementedError("Abstract method shall not be invoked!")

    @abstractmethod
    def is_eligible(self, task: str) -> bool:
        raise NotImplementedError("Abstract method shall not be invoked!")
