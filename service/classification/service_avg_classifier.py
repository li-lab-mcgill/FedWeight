from typing import Dict

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from service.classification.service_abstract_classifier import AbstractClassifierService

from utils.utils_eval import EvaluationUtils
from utils.constants import *

from model.result import Result

import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from scipy.cluster.hierarchy import linkage

import copy

import umap.umap_ as umap
from matplotlib.lines import Line2D

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import matplotlib.pyplot as plt


class FedWeightAvgClassifierService(AbstractClassifierService):
    """ Public methods """

    def run_classification(self, **kwargs) -> Result:

        result = Result()

        model = kwargs.get("model")
        opt = kwargs.get("opt")
        x = kwargs.get("x")
        y = kwargs.get("y")
        batch_size = kwargs.get("batch_size")
        split = kwargs.get("split")
        reweight = kwargs.get("reweight")
        focal_alpha = kwargs.get("focal_alpha")
        focal_gamma = kwargs.get("focal_gamma")
        linear_regression = kwargs.get("linear_regression")
        unsupervised = kwargs.get("unsupervised")

        need_plot = kwargs.get("need_plot")
        PID = kwargs.get("PID")

        assert model is not None
        assert x is not None
        assert y is not None
        assert split is not None

        # Train
        if split == 'train':

            global_weights = copy.deepcopy(model.state_dict())

            assert opt is not None
            assert batch_size is not None

            if batch_size <= 0 or batch_size > len(y):
                raise ValueError(
                    "Batch size must be larger than 0 and smaller than sample size")

            torch.set_grad_enabled(True)
            model.train()

            # Train
            N = len(y)
            B = batch_size
            nsteps = N // B

            for step in range(nsteps):
                # fetch the next batch of data
                x_batch = Variable(x[step * B: step * B + B])  # batch_size x D
                y_batch = Variable(y[step * B: step * B + B])  # batch_size x 1
                assert len(x_batch) == B
                assert len(y_batch) == B

                # Loss
                if unsupervised:
                    # batch_size, for weighted backprop
                    recon_loss_sample, kld_sample, _ = model(x_batch)
                    loss_sample = recon_loss_sample + kld_sample
                    # scaler, for unweighted backprop
                    loss_batch = torch.mean(loss_sample)

                else:
                    pred_batch = model(x_batch)  # batch_size x 1
                    assert len(pred_batch) == B
                    if not linear_regression:
                        if focal_gamma is not None:
                            loss_sample = EvaluationUtils.mean_focal_loss(pred_batch, y_batch,
                                                                          focal_alpha, focal_gamma)  # batch_size x 1
                            # scaler, for unweighted backprop
                            loss_batch = torch.mean(loss_sample)
                            # batch_size, for weighted backprop
                            loss_sample = torch.mean(loss_sample, dim=1)

                        else:
                            loss_sample = EvaluationUtils.mean_bce(pred_batch, y_batch,
                                                                   reduction='none')  # batch_size x 1
                            # scaler, for unweighted backprop
                            loss_batch = torch.mean(loss_sample)
                            # batch_size, for weighted backprop
                            loss_sample = torch.mean(loss_sample, dim=1)
                    else:
                        loss_sample = EvaluationUtils.mean_mse(pred_batch, y_batch,
                                                               reduction='none')  # batch_size x 1
                        # scaler, for unweighted backprop
                        loss_batch = torch.mean(loss_sample)
                        # batch_size, for weighted backprop
                        loss_sample = torch.mean(loss_sample, dim=1)

                # Reweight
                if reweight is not None:

                    # batch_size x 1
                    reweight_batch = reweight[step * B: step * B + B]
                    assert len(loss_sample) == len(reweight_batch)
                    loss_reweight = torch.mean(
                        torch.mul(reweight_batch, loss_sample))

                    opt.zero_grad()
                    loss_reweight.backward()
                    opt.step()
                else:
                    opt.zero_grad()
                    loss_batch.backward()
                    opt.step()

            result.model_params = self._get_model_params(model)

        # Test
        torch.set_grad_enabled(False)
        model.eval()

        if unsupervised:

            recon_loss_sample, kld_sample, beta = model(x)
            loss_sample = recon_loss_sample + kld_sample
            loss = torch.mean(loss_sample)

            x_np = x.cpu().detach().numpy()
            x_bow = []
            for row in x_np:
                word_id = list(np.where(row == 1)[0])
                x_bow.append(word_id)

            result.loss = loss.item()
            result.acc = 0
            # result.roc = roc
            result.pr = 0
            result.f1_score = 0
            result.ari = 0

            # if target_binary is not None:
            #
            #     _, latent = model(unsupervised_x)
            #
            #     latent = latent.cpu().detach().numpy()
            #     ari_score = self._get_ari(latent, target_binary, PID, cur_seed)
            #     result.ari = ari_score
            # else:
            #     result.ari = 0

            beta_np = beta.cpu().detach().numpy()
            coherence = EvaluationUtils.get_topic_coherence(beta_np, x_bow)
            diversity = EvaluationUtils.get_topic_diversity(beta_np, 10)
            quality = coherence * diversity

            result.coherence = coherence
            result.diversity = diversity
            result.quality = quality

            if need_plot:
                self._plot_beta(beta_np, PID)

        else:
            pred = model(x)
            y = y.reshape(-1, 1)
            pred = pred.reshape(-1, 1)
            if not linear_regression:
                loss = EvaluationUtils.mean_bce(pred, y)
                acc = EvaluationUtils.mean_accuracy(pred, y)
                # roc = EvaluationUtils.mean_roc_auc(pred, y)
                pr = EvaluationUtils.mean_pr_auc(pred, y)
                f1_score = EvaluationUtils.f1_score(pred, y)

                result.loss = loss.item()
                result.acc = acc.item()
                # result.roc = roc
                result.pr = pr
                result.f1_score = f1_score
                result.ari = 0
                result.coherence = 0
                result.diversity = 0
            else:
                loss = EvaluationUtils.mean_mse(pred, y)

                result.loss = loss.item()
                result.acc = 0
                # result.roc = 0
                result.pr = 0
                result.f1_score = 0
                result.ari = 0
                result.coherence = 0
                result.diversity = 0

        result.sample_size = len(y)

        return result

    def is_eligible(self, type: str) -> bool:
        return type == FED_WEIGHT_METHOD_AVG

    """ Private methods """

    def _get_model_params(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return copy.deepcopy(model.state_dict())

    def _get_ari(self, latent, target_binary, PID, cur_seed):

        # y_np = y.cpu().detach().numpy()
        # unsupervised_x = x.cpu().detach().numpy()

        reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.25,
                            random_state=np.random.RandomState(25),
                            transform_seed=np.random.RandomState(25))
        principal_components = reducer.fit_transform(latent)
        print(principal_components.shape)

        kmeans = KMeans(n_clusters=2, random_state=42)
        kmeans.fit(principal_components)
        kmeans_labels = kmeans.labels_

        ari_score_hospital = adjusted_rand_score(target_binary[:, 0], kmeans_labels)
        print(f'Adjusted Rand Index (ARI) for Hospitals: {ari_score_hospital}')

        # target_binary = target_binary[:, None]

        if cur_seed == 0:
            final_np = np.hstack((principal_components, target_binary))

            scatter_rows_0 = final_np[np.where(final_np[:, 2] == 1.0)]
            scatter_rows_1 = final_np[np.where(final_np[:, 2] == 0.0)]

            # Plot UMAP
            plt.clf()
            plt.figure(figsize=(6, 5))
            plt.scatter(scatter_rows_1[:, 0], scatter_rows_1[:, 1], color='#1268fd', label='Source', s=0.2, alpha=0.2)
            plt.scatter(scatter_rows_0[:, 0], scatter_rows_0[:, 1], color='#ff5c7c', label='Target', s=0.2, alpha=0.5)

            plt.title(f'UMAP Federated Model Latent')

            legend_elements = []
            legend_element_0 = Line2D([0], [0], marker='o', color='w', markerfacecolor='#1268fd', markersize=8,
                                      label="Source")
            legend_element_1 = Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff5c7c', markersize=8,
                                      label="Target")
            legend_elements.append(legend_element_0)
            legend_elements.append(legend_element_1)

            plt.legend(title="Hospitals", handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(f'PID: {PID} - UMAP Federated Model Latent.png', bbox_inches='tight')
            plt.close()

            # Create a seaborn clustermap
            plt.clf()
            row_clusters = linkage(latent, method='ward')
            col_clusters = linkage(latent.T, method='ward')
            cmap = sns.diverging_palette(260, 350, as_cmap=True)
            g = sns.clustermap(latent, row_linkage=row_clusters, col_linkage=col_clusters, figsize=(5, 6),
                               yticklabels=False, cmap=cmap, center=0,
                               cbar_kws={'orientation': 'horizontal', 'pad': 0.1, 'shrink': 0.6},
                               cbar_pos=(0.45, -0.05, 0.3, 0.02))

            g.fig.suptitle(f'Heatmap Federated Model Latent',
                           fontsize=12, x=0.6, y=1.02)
            g.ax_heatmap.set_xlabel('Latent Dimension')
            g.ax_heatmap.set_ylabel('Patients')

            plt.savefig(f'PID: {PID} - Heatmap Federated Model Latent.png', bbox_inches='tight')
            plt.close()

            np.savetxt(f'PID: {PID} - Numpy Federated Model Latent.txt', latent)

        return ari_score_hospital

    def _plot_beta(self, beta, PID):
        plt.figure(figsize=(8, 6))
        sns.heatmap(beta, cmap="viridis", annot=False, cbar=True)

        plt.clf()
        plt.title("Heatmap Topic-word Distribution")
        plt.xlabel("Features")
        plt.ylabel("Topics")

        plt.savefig(f'PID: {PID} - Heatmap Topic-word Distribution.png', bbox_inches='tight')
        plt.close()

        np.savetxt(f'PID: {PID} - Numpy Topic-word Distribution.txt', beta)
