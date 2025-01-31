from model.model_classifier import ModelBinaryClassification
from model.model_lstm import LSTM
from model.model_etm import ETM
from model.model_made import MADE
from model.model_vae import VAE
from model.model_vqvae import VQVAE
from model.result import Result
from utils.utils_log import LogUtils
from utils.utils_pid import PidUtils
from utils.utils_eval import EvaluationUtils
from utils.constants import *

import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import resample
from scipy.sparse import csr_matrix

import numpy as np
import torch
import copy


class Client:

    def __init__(self, hospital_id, cls_x, cls_y, density_x,
                 made_service, vae_service, vqvae_service, classifier_service,
                 data_split_service, plot_service,
                 hyperparam_federated,
                 hyperparam_made,
                 hyperparam_vae,
                 topics,
                 hyperparam_focal,
                 device,
                 algorithm,
                 output_path,
                 task,
                 reweight_phi=None,  # Weight log-likelihood
                 reweight_lambda=None,  # Weight log-likelihood
                 bias_init_prior_prob=None,
                 linear_regression=False,
                 unsupervised=False,
                 test_with_bootstrap=False,
                 dedicated_hospital_ids=None,
                 run_with_fl=True,
                 density_estimator='vae'):

        self._hospital_id = hospital_id
        self._dedicated_hospital_ids = dedicated_hospital_ids
        self._run_with_fl = run_with_fl

        self._cls_x = np.asarray(cls_x)
        self._cls_x = self._cls_x.astype(np.float32)
        self._cls_x = torch.Tensor(self._cls_x).to(device)

        self._cls_y = np.asarray(cls_y)
        self._cls_y = self._cls_y.astype(np.float32)
        # self._cls_y = self._cls_y.reshape(-1, 1)
        self._cls_y = torch.Tensor(self._cls_y).to(device)

        self._density_x = np.asarray(density_x)
        self._density_x = self._density_x.astype(np.float32)
        self._density_x = torch.Tensor(self._density_x).to(device)

        # Debug
        # torch.manual_seed(42)
        # random_indices = torch.randperm(len(self._cls_x))[:2000]
        # self._cls_x = self._cls_x[random_indices]
        # self._cls_y = self._cls_y[random_indices]

        # Service
        self._made_service = made_service
        self._vae_service = vae_service
        self._vqvae_service = vqvae_service
        self._classifier_service = classifier_service
        self._data_split_service = data_split_service
        self._plot_service = plot_service

        # FL
        self._local_epochs = hyperparam_federated.local_epochs
        self._learning_rate = hyperparam_federated.learning_rate
        self._val_size = hyperparam_federated.val_size
        self._weight_decay = hyperparam_federated.weight_decay
        self._total_feature = hyperparam_federated.total_feature
        self._fl_hiddens = hyperparam_federated.fl_hiddens
        self._fl_batch_size = hyperparam_federated.batch_size

        # MADE
        self._made_epochs = hyperparam_made.made_epochs
        self._made_hiddens = hyperparam_made.made_hiddens
        self._num_masks = hyperparam_made.num_masks
        self._samples = hyperparam_made.samples
        self._resample_every = hyperparam_made.resample_every
        self._natural_ordering = hyperparam_made.natural_ordering
        self._made_learning_rate = hyperparam_made.made_learning_rate
        self._made_weight_decay = hyperparam_made._made_weight_decay
        self._made_batch_size = hyperparam_made.batch_size

        # VAE
        self._vae_epochs = hyperparam_vae.vae_epochs
        self._vae_latent_dim = hyperparam_vae.vae_latent_dim
        self._vae_hiddens = hyperparam_vae.vae_hiddens
        self._vae_learning_rate = hyperparam_vae.vae_learning_rate
        self._vae_weight_decay = hyperparam_vae.vae_weight_decay
        self._vae_batch_size = hyperparam_vae.batch_size

        # LDA
        self._lda_topics = topics

        # Reweight
        self._reweight_phi = reweight_phi if reweight_phi is not None else 1.0
        self._reweight_lambda = reweight_lambda if reweight_lambda is not None else 1.0

        # Focal loss
        self._focal_alpha = hyperparam_focal.focal_alpha
        self._focal_gamma = hyperparam_focal.focal_gamma
        self._bias_init_prior_prob = bias_init_prior_prob

        self._algorithm = algorithm
        self._task = task
        self._device = device
        self._output_path = output_path
        self._linear_regression = linear_regression
        self._unsupervised = unsupervised
        self._test_with_bootstrap = test_with_bootstrap
        self._density_estimator = density_estimator

    """ MADE """

    def train_density_estimator(self):

        if self._density_estimator == 'made':
            self._train_made()
        elif self._density_estimator == 'vae':
            self._train_vae()
        elif self._density_estimator == 'vqvae':
            self._train_vq_vae()
        elif self._density_estimator == 'lda':
            self._train_lda()
        else:
            raise NotImplementedError("Unrecognized density estimator")

    def _train_made(self):

        # Train MADE
        hidden_list = list(map(int, self._made_hiddens.split(',')))
        made = MADE(self._density_x.size(1), hidden_list,
                    self._density_x.size(1), num_masks=self._num_masks,
                    natural_ordering=self._natural_ordering)
        made.to(self._device)
        made_opt = torch.optim.Adam(made.parameters(),
                                    lr=self._made_learning_rate,
                                    weight_decay=self._made_weight_decay)
        made_scheduler = torch.optim.lr_scheduler.StepLR(
            made_opt, step_size=45, gamma=0.1)

        made_train_hist = []

        epochs_without_improvement = 0
        best_train_loss = float('inf')
        for epoch in range(self._made_epochs):
            made_train_loss, _ = self._made_service.run_made(made, made_opt,
                                                             self._density_x, None,
                                                             'train', self._made_batch_size,
                                                             self._samples,
                                                             self._resample_every,
                                                             self._device)

            made_scheduler.step()
            made_train_hist.append(made_train_loss)

            if made_train_loss < best_train_loss:
                best_train_loss = made_train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # if epochs_without_improvement >= 5:
            #     LogUtils.instance().log_info(f'Early stopping after {epoch} rounds without improvement')
            #     break

            LogUtils.instance().log_info(
                "Train MADE on source hospital {} - epoch: {}, train loss: {:.5f}".format(self._hospital_id, epoch,
                                                                                          made_train_loss),
                type=LOGGER_MADE)

        # Plot
        pid = PidUtils.instance().get_pid()
        title = "Source hospital {} MADE loss".format(self._hospital_id)
        file_name = "{}/PID: {} - source_hospital: {} made_loss.png".format(
            self._output_path, pid, self._hospital_id)

        plt.clf()
        plt.plot(made_train_hist, label="train loss")
        plt.title(title)
        plt.legend()
        plt.grid(0.5)
        plt.savefig(file_name)
        plt.close()

        # Source prob estimator (q0)
        _, self._source_loss_for_samples = self._made_service.run_made(made, None,
                                                                       None, self._density_x, 'test',
                                                                       self._made_batch_size,
                                                                       self._samples, self._resample_every,
                                                                       self._device)

    def _train_vae(self):

        # Train VAE
        vae = VAE(self._density_x.size(1),
                  self._vae_hiddens,
                  self._vae_latent_dim)
        vae.to(self._device)
        vae_opt = torch.optim.Adam(vae.parameters(),
                                   lr=self._vae_learning_rate,
                                   weight_decay=self._vae_weight_decay)
        vae_scheduler = torch.optim.lr_scheduler.StepLR(
            vae_opt, step_size=45, gamma=0.1)

        vae_train_hist = []

        epochs_without_improvement = 0
        best_train_loss = float('inf')
        for epoch in range(self._vae_epochs):
            beta = 0.001 * epoch
            beta = beta if beta < 1.0 else 1.0
            vae_train_loss, _, recon_loss, kl_loss = self._vae_service.run_vae(vae, vae_opt,
                                                                               self._density_x, None,
                                                                               'train', self._vae_batch_size,
                                                                               beta, self._device)

            vae_scheduler.step()
            vae_train_hist.append(vae_train_loss)

            if vae_train_loss < best_train_loss:
                best_train_loss = vae_train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # if epochs_without_improvement >= 10:
            #     LogUtils.instance().log_info(f'Early stopping after {epoch} rounds without improvement')
            #     break

            LogUtils.instance().log_info(
                "Train VAE on source hospital {} - epoch: {}, train loss: {:.5f}, recon_loss: {:.5f}, kl_loss: {:.5f}".format(
                    self._hospital_id, epoch, vae_train_loss, recon_loss, kl_loss), type=LOGGER_MADE)

        # Plot
        pid = PidUtils.instance().get_pid()
        title = "Source hospital {} VAE loss".format(self._hospital_id)
        file_name = "{}/PID: {} - source_hospital: {} vae_loss.png".format(
            self._output_path, pid, self._hospital_id)

        plt.clf()
        plt.plot(vae_train_hist, label="train loss")
        plt.title(title)
        plt.legend()
        plt.grid(0.5)
        plt.savefig(file_name)
        plt.close()

        # Source prob estimator (q0)
        _, self._source_loss_for_samples, _, _ = self._vae_service.run_vae(vae, None,
                                                                           None, self._density_x,
                                                                           'test', self._vae_batch_size,
                                                                           1.0, self._device)

    def _train_vq_vae(self):

        vqvae = VQVAE(self._density_x.size(1),
                      self._vae_hiddens,
                      self._vae_latent_dim,
                      0.25,
                      self._device)

        vqvae.to(self._device)
        vqvae_opt = torch.optim.Adam(vqvae.parameters(),
                                     lr=self._vae_learning_rate,
                                     weight_decay=self._vae_weight_decay)
        vqvae_scheduler = torch.optim.lr_scheduler.StepLR(
            vqvae_opt, step_size=45, gamma=0.1)

        vqvae_train_hist = []
        epochs_without_improvement = 0
        best_train_loss = float('inf')
        for epoch in range(self._vae_epochs):
            vqvae_train_loss, _ = self._vqvae_service.run_vqvae(vqvae, vqvae_opt,
                                                                self._density_x, None,
                                                                'train', self._vae_batch_size,
                                                                self._device)

            vqvae_scheduler.step()
            vqvae_train_hist.append(vqvae_train_loss)

            if vqvae_train_loss < best_train_loss:
                best_train_loss = vqvae_train_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # if epochs_without_improvement >= 10:
            #     LogUtils.instance().log_info(f'Early stopping after {epoch} rounds without improvement')
            #     break

            LogUtils.instance().log_info(
                "Train VQVAE on source hospital {} - epoch: {}, train loss: {:.5f}".format(self._hospital_id, epoch,
                                                                                           vqvae_train_loss),
                type=LOGGER_MADE)

        # Plot
        pid = PidUtils.instance().get_pid()
        title = "Source hospital {} VQVAE loss".format(self._hospital_id)
        file_name = "{}/PID: {} - source_hospital: {} vqvae_loss.png".format(
            self._output_path, pid, self._hospital_id)

        plt.clf()
        plt.plot(vqvae_train_hist, label="train loss")
        plt.title(title)
        plt.legend()
        plt.grid(0.5)
        plt.savefig(file_name)
        plt.close()

        # Source prob estimator (q0)
        _, self._source_loss_for_samples = self._vqvae_service.run_vqvae(vqvae, None,
                                                                         None, self._density_x,
                                                                         'test', self._vae_batch_size,
                                                                         self._device)

    def _train_lda(self):

        lda_model = LatentDirichletAllocation(n_components=self._lda_topics,
                                              random_state=42)
        x_train = self._density_x.cpu().detach().numpy().copy()
        x_train = csr_matrix(x_train)
        lda_model.fit(x_train)

        perplexity = lda_model.perplexity(x_train)

        LogUtils.instance().log_info(
            "Train LDA on source hospital {} perplexity: {:.5f}".format(self._hospital_id, perplexity),
            type=LOGGER_MADE)

        document_topic_distribution = lda_model.transform(x_train)

        topic_word_distribution = lda_model.components_ / \
                                  lda_model.components_.sum(axis=1)[:, np.newaxis]
        pred = document_topic_distribution @ topic_word_distribution  # N x D

        pred = torch.Tensor(pred).to(self._device)
        # loss_sample = EvaluationUtils.mean_ce(
        #     pred, self._density_x, reduction='none')  # N x 1
        # num_drugs_taken = torch.sum(self._density_x, dim=1)  # N x 1
        # self._source_loss_for_samples = loss_sample / num_drugs_taken  # N x 1

        pred_l = torch.sigmoid(pred)
        loss_sample = EvaluationUtils.mean_bce(pred_l, self._density_x, reduction='none')
        self._source_loss_for_samples = torch.mean(loss_sample, dim=1)

    """ Classification """

    def train_cls(self, round, global_model, target_estimator=None) -> Result:

        # Reweight
        reweight = None
        if target_estimator is not None:
            # Target prob estimator (q1)
            if self._density_estimator == 'made':
                _, target_loss_for_samples = self._made_service.run_made(target_estimator, None,
                                                                         None, self._density_x, 'test',
                                                                         self._made_batch_size,
                                                                         self._samples, self._resample_every,
                                                                         self._device)
            elif self._density_estimator == 'vae':
                _, target_loss_for_samples, _, _ = self._vae_service.run_vae(target_estimator, None,
                                                                             None, self._density_x,
                                                                             'test', self._vae_batch_size,
                                                                             1.0, self._device)
            elif self._density_estimator == 'vqvae':
                _, target_loss_for_samples = self._vqvae_service.run_vqvae(target_estimator, None,
                                                                           None, self._density_x,
                                                                           'test', self._vae_batch_size,
                                                                           self._device)
            elif self._density_estimator == 'lda':

                x_train = self._density_x.cpu().detach().numpy().copy()
                x_train = csr_matrix(x_train)
                document_topic_distribution = target_estimator.transform(
                    x_train)

                topic_word_distribution = target_estimator.components_ / \
                                          target_estimator.components_.sum(axis=1)[:, np.newaxis]
                pred = document_topic_distribution @ topic_word_distribution  # N x D

                pred = torch.Tensor(pred).to(self._device)
                # loss_sample = EvaluationUtils.mean_ce(
                #     pred, self._density_x, reduction='none')  # N x 1
                # num_drugs_taken = torch.sum(self._density_x, dim=1)  # N x 1
                # target_loss_for_samples = loss_sample / num_drugs_taken  # N x 1

                pred_l = torch.sigmoid(pred)
                loss_sample = EvaluationUtils.mean_bce(pred_l, self._density_x, reduction='none')
                target_loss_for_samples = torch.mean(loss_sample, dim=1)

            else:
                raise NotImplementedError("Unrecognized density estimator")
            # LogUtils.instance().log_info(
            #     "Target loss for samples: {}".format(target_loss_for_samples))

            loss_diff = self._source_loss_for_samples - target_loss_for_samples  # N x 1
            # reweight = p_t(x) / p_s(x) = exp(-target_loss) / exp(-source_loss) = exp(source_loss - target_loss)
            reweight = torch.exp(loss_diff)  # N x 1

            # Example arrays A and B
            # Sort array A and get sorted indices
            # A = torch.sum(self._cls_x, dim=1).numpy()
            # B = loss_diff.numpy()

            # argsort returns indices of sorted elements
            # sorted_indices = np.argsort(A)[::-1]

            # # Use sorted indices to retrieve corresponding values from array B
            # sorted_A = A[sorted_indices]
            # sorted_B = B[sorted_indices]

            # print(sorted_A.shape)
            # print(sorted_B.shape)

            # # Create dot plot
            # plt.clf()
            # plt.scatter(sorted_A, sorted_B)
            # plt.xlabel('Number of drugs taken by patients in Hospital 110')
            # plt.ylabel('Source loss - target loss + source KL - target KL')
            # plt.title(
            #     'Number of drugs taken by patients in Hospital 110 and loss KL difference')
            # plt.grid(True)
            # plt.show()

            # self._debug_loss_diff_plot(loss_diff)
            # self._debug_reweight_plot(reweight, after=False)
            # reweight = self._reweight_phi * \
            #            torch.pow(reweight, self._reweight_lambda)
            # self._debug_reweight_plot(reweight, after=True)

        # self._debug_source_made(self._made_hiddens, target_made)

        # Update local weights
        global_weights = global_model.state_dict()
        global_weights_cp = copy.deepcopy(global_weights)

        if self._task == DEATH or self._task == LENGTH:
            local_model = ModelBinaryClassification(self._total_feature,
                                                    self._fl_hiddens,
                                                    self._bias_init_prior_prob,
                                                    linear_regression=self._linear_regression)
        elif self._task == UNSUPERVISED:
            local_model = ETM(self._total_feature,
                              dedicated_hospital_ids=self._dedicated_hospital_ids,
                              run_with_fl=self._run_with_fl)
        else:
            local_model = LSTM(self._total_feature)

        local_model = local_model.to(self._device)
        local_model.load_state_dict(global_weights_cp)
        opt = torch.optim.Adam(local_model.parameters(),
                               lr=self._learning_rate,
                               weight_decay=self._weight_decay)

        # if round == 0:
        #     LogUtils.instance().log_info(
        #         "Initial model weights: {}".format(local_model.state_dict()))

        # Train
        for _ in range(self._local_epochs):
            result = self._classifier_service.run_classification(model=local_model,
                                                                 opt=opt,
                                                                 x=self._cls_x,
                                                                 y=self._cls_y,
                                                                 batch_size=self._fl_batch_size,
                                                                 split='train',
                                                                 reweight=reweight,
                                                                 focal_alpha=self._focal_alpha,
                                                                 focal_gamma=self._focal_gamma,
                                                                 linear_regression=self._linear_regression,
                                                                 unsupervised=self._unsupervised)

        return result

    def get_data(self):
        return self._cls_x, self._cls_y

    """ Debug Plot """

    def _debug_loss_diff_plot(self, loss_diff):

        hist, bin_edges = np.histogram(loss_diff.cpu().detach().numpy().copy(),
                                       bins=20)
        labels = []
        for b0, b1 in zip(bin_edges, bin_edges[1:]):
            labels.append("{:.2f} ~ {:.2f}".format(b0, b1))

        plt.clf()
        plt.title("Loss difference: Source VQVAE loss - target VQVAE loss")
        plt.barh(labels, hist)
        plt.tight_layout()
        plt.ylabel("Loss difference")
        plt.xlabel("Samples count")
        # for i, v in enumerate(hist):
        #     plt.text(v + 3, i-0.25, str(v), color='red', fontweight='bold')
        plt.savefig("{}/{}_loss_bin.png".format(self._output_path, self._hospital_id),
                    bbox_inches='tight')
        plt.close()

    def _debug_reweight_plot(self, reweight, after=False):

        hist, bin_edges = np.histogram(reweight.cpu().detach().numpy().copy(),
                                       bins=20)
        labels = []
        for b0, b1 in zip(bin_edges, bin_edges[1:]):
            labels.append("{:.2f} ~ {:.2f}".format(b0, b1))

        plt.clf()
        if after:
            plt.title(
                "Reweight V = exp(Source VQVAE loss - target VQVAE loss) - after")
        else:
            plt.title(
                "Reweight V = exp(Source VQVAE loss - target VQVAE loss) - before")
        plt.barh(labels, hist)
        plt.tight_layout()
        plt.ylabel("Reweight")
        plt.xlabel("Samples count")
        # for i, v in enumerate(hist):
        #     plt.text(v + 3, i-0.25, str(v), color='red', fontweight='bold')
        if after:
            plt.savefig("{}/reweight_bin_after_hospital_{}.png".format(self._output_path, self._hospital_id),
                        bbox_inches='tight')
        else:
            plt.savefig("{}/reweight_bin_before_hospital_{}.png".format(self._output_path, self._hospital_id),
                        bbox_inches='tight')
        plt.close()
