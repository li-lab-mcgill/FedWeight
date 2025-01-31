from typing import Tuple, Dict
from collections import OrderedDict
import torch
import torch.nn as nn
import copy

from service.federated.service_abstract_federated import AbstractFederatedService
from service.classification.service_abstract_classifier import AbstractClassifierService
from model.result import Result
from utils.constants import *


class FedWeightAvgService(AbstractFederatedService):

    """ Initialize """

    def __init__(self, classifier_service: AbstractClassifierService) -> None:
        self._classifier_service = classifier_service

    """ Public methods """

    def run_fed(self, round: int,
                global_model: nn.Module,
                global_optimizer: torch.optim,
                source_hospitals: list,
                x_target_val: torch.Tensor,
                y_target_val: torch.Tensor,
                target_estimator: nn.Module = None,
                linear_regression: bool = False,
                unsupervised: bool = False) -> Tuple[Result, Result]:

        total_weights = []
        total_sizes = []
        total_train_loss = []
        total_train_acc = []
        total_train_roc = []
        total_train_pr = []

        for id, client in enumerate(source_hospitals):

            result = client.train_cls(round,
                                      global_model,
                                      target_estimator)

            total_weights.append(result.model_params)
            total_sizes.append(result.sample_size)
            total_train_loss.append(result.loss)
            total_train_acc.append(result.acc)
            # total_train_roc.append(result.roc)
            total_train_pr.append(result.pr)
        
        # Average weights of all source hospitals
        avg_weights = self._average_weights(total_weights,
                                            total_sizes)
        avg_weights_cp = copy.deepcopy(avg_weights)
        global_model.load_state_dict(avg_weights_cp)

        train_loss = self._average_validation(total_train_loss, total_sizes)
        train_acc = self._average_validation(total_train_acc, total_sizes)
        # train_roc = self._average_validation(total_train_roc, total_sizes)
        train_pr = self._average_validation(total_train_pr, total_sizes)

        train_result = Result()

        train_result.loss = train_loss
        train_result.acc = train_acc
        # train_result.roc = train_roc
        train_result.pr = train_pr

        # Validate on target
        val_result = self._classifier_service.run_classification(model=global_model,
                                                                 x=x_target_val,
                                                                 y=y_target_val,
                                                                 split='test',
                                                                 linear_regression=linear_regression,
                                                                 unsupervised=unsupervised)

        return (train_result, val_result)

    def is_eligible(self, type: str) -> bool:
        return type == FED_WEIGHT_METHOD_AVG

    """ Private methods """

    def _average_weights(self, w: list,
                         sizes: list) -> Dict[str, torch.Tensor]:
        """
        Returns the average of the weights.
        """
        result: Dict[str, torch.Tensor] = OrderedDict()
        w_cp = copy.deepcopy(w[0])
        for key in w_cp.keys():
            for i in range(len(w)):
                if result.get(key) is None:
                    result[key] = 0.0
                factor = sizes[i] / sum(sizes)
                result[key] += (w[i][key] * factor)
        return result

    def _average_validation(self, v: list, sizes: list) -> int:
        """
        Returns the average of the validation history.
        """
        result = 0
        for i, val in enumerate(v):
            factor = sizes[i] / sum(sizes)
            result += factor * val
        return result
