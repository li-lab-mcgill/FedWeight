from abc import ABC, abstractmethod
from typing import Tuple

from service.classification.service_abstract_classifier import AbstractClassifierService

import torch
import torch.nn as nn

from model.result import Result


class AbstractFederatedService(ABC):

    @abstractmethod
    def __init__(self, classifier_service: AbstractClassifierService) -> None:
        raise NotImplementedError("Abstract method shall not be invoked!")

    @abstractmethod
    def run_fed(self, round: int,
                global_model: nn.Module,
                global_optimizer: torch.optim,
                source_hospitals: list,
                x_target_val: torch.Tensor,
                y_target_val: torch.Tensor,
                target_made: nn.Module = None,
                linear_regression: bool = False) -> Tuple[Result, Result]:
        raise NotImplementedError("Abstract method shall not be invoked!")

    @abstractmethod
    def is_eligible(self, type: str) -> bool:
        raise NotImplementedError("Abstract method shall not be invoked!")
