from typing import Tuple
import torch
import torch.nn as nn

from service.federated.service_abstract_federated import AbstractFederatedService
from service.classification.service_abstract_classifier import AbstractClassifierService
from model.result import Result
from utils.constants import *


class FedWeightSgdService(AbstractFederatedService):

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
                linear_regression: bool = False) -> Tuple[Result, Result]:

        total_grads = []
        total_sizes = []
        total_train_loss = []
        total_train_acc = []
        total_train_roc = []
        total_train_pr = []

        for id, client in enumerate(source_hospitals):

            result = client.train_cls(round,
                                      global_model,
                                      target_estimator)

            total_grads.append(result.model_grads)
            total_sizes.append(result.sample_size)
            total_train_loss.append(result.loss)
            total_train_acc.append(result.acc)
            total_train_roc.append(result.roc)
            total_train_pr.append(result.pr)

        train_loss = self._average_validation(total_train_loss, total_sizes)
        train_acc = self._average_validation(total_train_acc, total_sizes)
        train_roc = self._average_validation(total_train_roc, total_sizes)
        train_pr = self._average_validation(total_train_pr, total_sizes)

        train_result = Result()

        train_result.loss = train_loss
        train_result.acc = train_acc
        train_result.roc = train_roc
        train_result.pr = train_pr

        # Update global model using FedSGD
        self._update_global_model(global_model,
                                  global_optimizer,
                                  total_grads,
                                  total_sizes)

        val_result = self._classifier_service.run_classification(model=global_model,
                                                                 x=x_target_val,
                                                                 y=y_target_val,
                                                                 split='test',
                                                                 linear_regression=linear_regression)

        return (train_result, val_result)

    def is_eligible(self, type: str) -> bool:
        return type == FED_WEIGHT_METHOD_SGD

    """ Private methods """

    def _average_validation(self, v: list, sizes: list) -> int:
        """
        Returns the average of the validation history.
        """
        result = 0
        for i, val in enumerate(v):
            factor = sizes[i] / sum(sizes)
            result += factor * val
        return result

    def _update_global_model(self, global_model: nn.Module,
                             global_optimizer: torch.optim,
                             total_grads: list,
                             total_sizes: list):

        global_optimizer.zero_grad()

        model_params = list(global_model.parameters())
        param_gradients = [[] for _ in model_params]

        # Loop for each environment
        for env, env_grads in enumerate(total_grads):
            factor = total_sizes[env] / sum(total_sizes)
            for idx, grads in enumerate(param_gradients):
                grad = factor * env_grads[idx]  # 1 x D
                grads.append(grad)

        assert len(param_gradients) == len(model_params)

        for param, grads in zip(model_params, param_gradients):
            grads = torch.stack(grads, dim=0)  # n_clients x 1 x D
            avg_grad = torch.sum(grads, dim=0)  # 1 x D
            param.grad = avg_grad

        global_optimizer.step()
