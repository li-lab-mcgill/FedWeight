from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score
from model.model_focal import FocalLoss
import torch.nn as nn
import numpy as np


class EvaluationUtils:
    """ Utility class to evaluate models """

    @staticmethod
    def mean_bce(pred, y, reduction='mean'):
        criterion = nn.BCELoss(reduction=reduction)
        return criterion(pred, y)

    @staticmethod
    def mean_ce(pred, y, reduction='mean'):
        criterion = nn.CrossEntropyLoss(reduction=reduction)
        return criterion(pred, y)

    @staticmethod
    def mean_mse(pred, y, reduction='mean'):
        criterion = nn.MSELoss(reduction=reduction)
        return criterion(pred, y)

    @staticmethod
    def mean_focal_loss(pred, y, alpha, gamma):
        criterion = FocalLoss(alpha, gamma)
        return criterion(pred, y)

    @staticmethod
    def mean_accuracy(pred, y):
        pred = (pred > 0.5).float()
        return ((pred - y).abs() < 1e-2).float().mean()

    @staticmethod
    def mean_roc_auc(pred, y):
        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        return roc_auc_score(y, pred)

    @staticmethod
    def mean_pr_auc(pred, y):
        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        precision, recall, _ = precision_recall_curve(y, pred)
        return auc(recall, precision)

    @staticmethod
    def f1_score(pred, y):
        y = y.detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        pred_labels = np.where(pred >= 0.5, 1, 0)
        return f1_score(y, pred_labels)

    @staticmethod
    def get_topic_diversity(beta, topk):
        num_topics = beta.shape[0]
        list_w = np.zeros((num_topics, topk))
        for k in range(num_topics):
            idx = beta[k, :].argsort()[-topk:][::-1]
            list_w[k, :] = idx
        n_unique = len(np.unique(list_w))
        TD = n_unique / (topk * num_topics)
        return TD

    @staticmethod
    def get_topic_coherence(beta, data):
        D = len(data)  ## number of docs...data is list of documents
        TC = []
        num_topics = len(beta)
        counter = 0
        for k in range(num_topics):
            top_10 = list(beta[k].argsort()[-11:][::-1])
            TC_k = 0
            for i, word in enumerate(top_10):
                # get D(w_i)
                D_wi = EvaluationUtils.get_document_frequency(data, word)
                j = i + 1
                tmp = 0
                while j < len(top_10) and j > i:
                    # get D(w_j) and D(w_i, w_j)
                    D_wj, D_wi_wj = EvaluationUtils.get_document_frequency(data, word, top_10[j])
                    # get f(w_i, w_j)
                    if D_wi_wj == 0:
                        f_wi_wj = -1
                    else:
                        f_wi_wj = -1 + (np.log(D_wi) + np.log(D_wj) - 2.0 * np.log(D)) / (np.log(D_wi_wj) - np.log(D))
                    # update tmp:
                    tmp += f_wi_wj
                    j += 1
                    counter += 1
                # update TC_k
                TC_k += tmp
            TC.append(TC_k)
        TC = np.mean(TC) / counter
        TC = (TC + 1) / 2
        return TC

    @staticmethod
    def get_document_frequency(data, wi, wj=None):
        if wj is None:
            D_wi = 0
            for l in range(len(data)):
                doc = data[l]
                if wi in doc:
                    D_wi += 1
            return D_wi
        D_wj = 0
        D_wi_wj = 0
        for l in range(len(data)):
            doc = data[l]
            if wj in doc:
                D_wj += 1
                if wi in doc:
                    D_wi_wj += 1
        return D_wj, D_wi_wj
