from typing import List

import torch
import torch.nn as nn

from service.classification.service_abstract_classifier import AbstractClassifierService

from utils.utils_eval import EvaluationUtils
from utils.constants import *

from model.result import Result

import copy


class FedWeightSgdClassifierService(AbstractClassifierService):

    def run_classification(self, **kwargs) -> Result:

        result = Result()

        model = kwargs.get("model")
        opt = kwargs.get("opt")
        x = kwargs.get("x")
        y = kwargs.get("y")
        split = kwargs.get("split")
        reweight = kwargs.get("reweight")
        reweight_lambda = kwargs.get("reweight_lambda")
        focal_alpha = kwargs.get("focal_alpha")
        focal_gamma = kwargs.get("focal_gamma")

        reweight_lambda = reweight_lambda if reweight_lambda is not None else 1.0

        assert model is not None
        assert x is not None
        assert y is not None
        assert split is not None

        if split == 'train':
            assert opt is not None

        # enable/disable grad for efficiency of forwarding test batches
        torch.set_grad_enabled(split == 'train')
        model.train() if split == 'train' else model.eval()

        pred = model(x)

        if focal_gamma is not None:
            loss_sample = EvaluationUtils.mean_focal_loss(pred, y,
                                                          focal_alpha, focal_gamma)  # N x 1
            loss = torch.mean(loss_sample)  # scaler, for unweighted backprop
            loss_sample = loss_sample[:, 0]  # N, for weighted backprop

        else:
            loss_sample = EvaluationUtils.mean_bce(pred, y,
                                                   reduction='none')  # N x 1
            loss = torch.mean(loss_sample)  # scaler, for unweighted backprop
            loss_sample = loss_sample[:, 0]  # N, for weighted backprop

        acc = EvaluationUtils.mean_accuracy(pred, y)
        roc = EvaluationUtils.mean_roc_auc(pred, y)
        pr = EvaluationUtils.mean_pr_auc(pred, y)

        if split == 'train':
            # Reweight
            if reweight is not None:
                assert len(loss_sample) == len(reweight)
                reweight = torch.pow(reweight, reweight_lambda)
                loss = torch.mean(torch.mul(reweight, loss_sample))

            opt.zero_grad()
            loss.backward()

        result.loss = loss.item()
        result.acc = acc.item()
        # result.roc = roc
        result.pr = pr
        result.sample_size = len(y)
        result.model_grads = self._get_model_grads(model)

        return result

    def is_eligible(self, type: str) -> bool:
        return type == FED_WEIGHT_METHOD_SGD

    def _get_model_grads(self, model: nn.Module) -> List[torch.Tensor]:

        result: List[torch.Tensor] = []
        model_params = list(model.parameters())
        for model_param in model_params:
            grad = model_param.grad
            result.append(copy.deepcopy(grad))
        return result
