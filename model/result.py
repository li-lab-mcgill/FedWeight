from typing import Dict, List
import torch


class Result:
    """ Initialize """

    def __init__(self):
        self._loss = None
        self._acc = None
        self._roc = None
        self._pr = None
        self._ari = None
        self._coherence = None
        self._diversity = None
        self._quality = None
        self._f1_score = None
        self._loss_hist = None
        self._acc_hist = None
        self._roc_hist = None
        self._pr_hist = None
        self._coherence_hist = None
        self._diversity_hist = None
        self._quality_hist = None
        self._model_params = None
        self._model_grads = None
        self._sample_size = None

    """ Getters """

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def acc(self) -> float:
        return self._acc

    @property
    def roc(self) -> float:
        return self._roc

    @property
    def pr(self) -> float:
        return self._pr

    @property
    def ari(self) -> float:
        return self._ari

    @property
    def coherence(self) -> float:
        return self._coherence

    @property
    def diversity(self) -> float:
        return self._diversity

    @property
    def quality(self) -> float:
        return self._quality

    @property
    def f1_score(self) -> float:
        return self._f1_score

    @property
    def loss_hist(self) -> list:
        return self._loss_hist

    @property
    def acc_hist(self) -> list:
        return self._acc_hist

    @property
    def roc_hist(self) -> list:
        return self._roc_hist

    @property
    def pr_hist(self) -> list:
        return self._pr_hist

    @property
    def coherence_hist(self) -> list:
        return self._coherence_hist

    @property
    def diversity_hist(self) -> list:
        return self._diversity_hist

    @property
    def quality_hist(self) -> list:
        return self._quality_hist

    @property
    def model_params(self) -> Dict[str, torch.Tensor]:
        return self._model_params

    @property
    def model_grads(self) -> List[torch.Tensor]:
        return self._model_grads

    @property
    def sample_size(self) -> int:
        return self._sample_size

    """ Setters """

    @loss.setter
    def loss(self, loss: float):
        self._loss = loss

    @acc.setter
    def acc(self, acc: float):
        self._acc = acc

    @roc.setter
    def roc(self, roc: float):
        self._roc = roc

    @pr.setter
    def pr(self, pr: float):
        self._pr = pr

    @ari.setter
    def ari(self, ari: float):
        self._ari = ari

    @coherence.setter
    def coherence(self, coherence: float):
        self._coherence = coherence

    @diversity.setter
    def diversity(self, diversity: float):
        self._diversity = diversity

    @quality.setter
    def quality(self, quality: float):
        self._quality = quality

    @f1_score.setter
    def f1_score(self, f1_score: float):
        self._f1_score = f1_score

    @loss_hist.setter
    def loss_hist(self, loss_hist: list):
        self._loss_hist = loss_hist

    @acc_hist.setter
    def acc_hist(self, acc_hist: list):
        self._acc_hist = acc_hist

    @roc_hist.setter
    def roc_hist(self, roc_hist: list):
        self._roc_hist = roc_hist

    @pr_hist.setter
    def pr_hist(self, pr_hist: list):
        self._pr_hist = pr_hist

    @coherence_hist.setter
    def coherence_hist(self, coherence_hist: list):
        self._coherence_hist = coherence_hist

    @diversity_hist.setter
    def diversity_hist(self, diversity_hist: list):
        self._diversity_hist = diversity_hist

    @quality_hist.setter
    def quality_hist(self, quality_hist: list):
        self._quality_hist = quality_hist

    @model_params.setter
    def model_params(self, model_params: Dict[str, torch.Tensor]):
        self._model_params = model_params

    @model_grads.setter
    def model_grads(self, model_grads: List[torch.Tensor]):
        self._model_grads = model_grads

    @sample_size.setter
    def sample_size(self, sample_size: int):
        self._sample_size = sample_size
