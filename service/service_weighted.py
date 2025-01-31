import copy
import torch

from model.result import Result
from model.model_classifier import ModelBinaryClassification
from model.model_etm import ETM
from model.model_lstm import LSTM
from model.model_made import MADE
from model.model_vae import VAE
from model.model_vqvae import VQVAE
from utils.utils_log import LogUtils
from utils.utils_pid import PidUtils
from utils.constants import *

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.utils import resample
from scipy.sparse import csr_matrix

import matplotlib.pyplot as plt


class WeightedService:
    """ Initialize """

    def __init__(self, federated_service,
                 classifier_service,
                 made_service,
                 vae_service,
                 vqvae_service,
                 plot_service,
                 data_split_service,
                 device, output_path,
                 density_estimator='vae'):
        self._federated_service = federated_service
        self._classifier_service = classifier_service
        self._made_service = made_service
        self._vae_service = vae_service
        self._vqvae_service = vqvae_service
        self._plot_service = plot_service
        self._data_split_service = data_split_service
        self._device = device
        self._output_path = output_path
        self._density_estimator = density_estimator

    """ Weighted """

    def train_weighted(self, seed, target_hospital_id,
                       source_hospitals,
                       hyperparam_federated,
                       hyperparam_made,
                       hyperparam_vae,
                       topics,
                       bias_init_prior_prob,
                       x_target_train,
                       x_target_val, y_target_val,
                       x_target_test, y_target_test,
                       global_model_weights,
                       total_feature,
                       task,
                       test_with_bootstrap=False,
                       linear_regression=False,
                       dedicated_hospital_ids=None,
                       run_with_fl=True,
                       unsupervised=False):

        train_result_hist = Result()
        val_result_hist = Result()
        test_result_by_best_pr_model = Result()
        test_result_by_best_loss_model = Result()

        if task == DEATH or task == LENGTH:
            weighted_global_model = ModelBinaryClassification(total_feature,
                                                              hyperparam_federated.fl_hiddens,
                                                              bias_init_prior_prob,
                                                              linear_regression=linear_regression)
        elif task == UNSUPERVISED:
            weighted_global_model = ETM(total_feature,
                                        dedicated_hospital_ids=dedicated_hospital_ids,
                                        run_with_fl=run_with_fl)
        else:
            weighted_global_model = LSTM(total_feature)

        weighted_global_model = weighted_global_model.to(self._device)
        weighted_global_model.load_state_dict(
            copy.deepcopy(global_model_weights))

        weighted_global_optimizer = torch.optim.Adam(weighted_global_model.parameters(),
                                                     lr=hyperparam_federated.learning_rate,
                                                     weight_decay=hyperparam_federated.weight_decay)

        # Validation
        weighted_best_val_pr = float('-inf')
        weighted_best_val_pr_round = None
        weighted_best_model_weights_by_pr = None

        weighted_best_val_loss = float('inf')
        weighted_best_val_loss_round = None
        weighted_best_model_weights_by_loss = None

        weighted_val_loss_hist = []
        weighted_val_acc_hist = []
        # weighted_val_roc_hist = []
        weighted_val_pr_hist = []
        weighted_val_coherence_hist = []
        weighted_val_diversity_hist = []
        weighted_val_quality_hist = []

        val_result_hist.loss_hist = weighted_val_loss_hist
        val_result_hist.acc_hist = weighted_val_acc_hist
        # val_result_hist.roc_hist = weighted_val_roc_hist
        val_result_hist.pr_hist = weighted_val_pr_hist
        val_result_hist.coherence_hist = weighted_val_coherence_hist
        val_result_hist.diversity_hist = weighted_val_diversity_hist
        val_result_hist.quality_hist = weighted_val_quality_hist

        # Train
        weighted_train_loss_hist = []
        weighted_train_acc_hist = []
        # weighted_train_roc_hist = []
        weighted_train_pr_hist = []

        train_result_hist.loss_hist = weighted_train_loss_hist
        train_result_hist.acc_hist = weighted_train_acc_hist
        # train_result_hist.roc_hist = weighted_train_roc_hist
        train_result_hist.pr_hist = weighted_train_pr_hist

        # Debug
        # torch.manual_seed(42)
        # random_indices = torch.randperm(len(x_target_train))[:2000]
        # x_target_train = x_target_train[random_indices]

        if self._density_estimator == 'made':
            target_estimator = self._train_target_made(
                target_hospital_id, hyperparam_made, x_target_train)
        elif self._density_estimator == 'vae':
            target_estimator = self._train_target_vae(
                target_hospital_id, hyperparam_vae, x_target_train)
        elif self._density_estimator == 'vqvae':
            target_estimator = self._train_target_vqvae(
                target_hospital_id, hyperparam_vae, x_target_train)
        elif self._density_estimator == 'lda':
            target_estimator = self._train_target_lda(
                target_hospital_id, topics, x_target_train)

        epochs_without_improvement = 0
        for round in range(hyperparam_federated.total_round):

            avg_train, avg_val = self._federated_service.run_fed(round,
                                                                 weighted_global_model,
                                                                 weighted_global_optimizer,
                                                                 source_hospitals,
                                                                 x_target_val, y_target_val,
                                                                 target_estimator,
                                                                 linear_regression=linear_regression,
                                                                 unsupervised=unsupervised)
            train_loss = avg_train.loss
            train_acc = avg_train.acc
            # train_roc = avg_train.roc
            train_pr = avg_train.pr

            val_loss = avg_val.loss
            val_acc = avg_val.acc
            # val_roc = avg_val.roc
            val_pr = avg_val.pr
            # val_coherence = avg_val.coherence
            # val_diversity = avg_val.diversity
            # val_quality = avg_val.quality

            # LogUtils.instance().log_info("Seed: {} - [Weighted] Target hospital id: {}, round: {}, train: loss: {:.5f}, acc: {:.5f}, roc: {:.5f}, pr: {:.5f}, val: loss: {:.5f}, acc: {:.5f}, roc: {:.5f}, pr: {:.5f}".format(
            #     seed, target_hospital_id, round, train_loss, train_acc, train_roc, train_pr, val_loss, val_acc, val_roc, val_pr))
            LogUtils.instance().log_info(
                "Seed: {} - [Weighted] Target hospital id: {}, round: {}, train: loss: {:.5f}, acc: {:.5f}, pr: {:.5f}, val: loss: {:.5f}, acc: {:.5f}, pr: {:.5f}".format(
                    seed, target_hospital_id, round, train_loss, train_acc, train_pr, val_loss, val_acc, val_pr))

            weighted_val_loss_hist.append(val_loss)
            weighted_val_acc_hist.append(val_acc)
            # weighted_val_roc_hist.append(val_roc)
            weighted_val_pr_hist.append(val_pr)

            weighted_train_loss_hist.append(train_loss)
            weighted_train_acc_hist.append(train_acc)
            # weighted_train_roc_hist.append(train_roc)
            weighted_train_pr_hist.append(train_pr)

            if val_pr > weighted_best_val_pr:
                weighted_best_val_pr = val_pr
                weighted_best_val_pr_round = round
                model_weights = weighted_global_model.state_dict()
                weighted_best_model_weights_by_pr = copy.deepcopy(
                    model_weights)
                if not linear_regression and not unsupervised:
                    epochs_without_improvement = 0
            else:
                if not linear_regression and not unsupervised:
                    epochs_without_improvement += 1

            if val_loss < weighted_best_val_loss:
                weighted_best_val_loss = val_loss
                weighted_best_val_loss_round = round
                model_weights = weighted_global_model.state_dict()
                weighted_best_model_weights_by_loss = copy.deepcopy(
                    model_weights)
                if linear_regression or unsupervised:
                    epochs_without_improvement = 0
            else:
                if linear_regression or unsupervised:
                    epochs_without_improvement += 1

            if epochs_without_improvement >= 10:
                LogUtils.instance().log_info(f'Early stopping after {round} rounds without improvement')
                break

        LogUtils.instance().log_info(
            "Seed: {} - [Weighted] Target hospital id: {}, Best val pr: {:.5f}, best val pr round: {}, Best val loss: {:.5f}, best val loss round: {}".format(
                seed, target_hospital_id, weighted_best_val_pr, weighted_best_val_pr_round, weighted_best_val_loss,
                weighted_best_val_loss_round))

        if task == DEATH or task == LENGTH:
            weighted_best_model_by_pr = ModelBinaryClassification(total_feature,
                                                                  hyperparam_federated.fl_hiddens,
                                                                  bias_init_prior_prob,
                                                                  linear_regression=linear_regression)
        elif task == UNSUPERVISED:
            weighted_best_model_by_pr = ETM(total_feature,
                                            dedicated_hospital_ids=dedicated_hospital_ids,
                                            run_with_fl=run_with_fl)
        else:
            weighted_best_model_by_pr = LSTM(total_feature)

        weighted_best_model_by_pr = weighted_best_model_by_pr.to(self._device)
        weighted_best_model_by_pr.load_state_dict(
            weighted_best_model_weights_by_pr)

        if task == DEATH or task == LENGTH:
            weighted_best_model_by_loss = ModelBinaryClassification(total_feature,
                                                                    hyperparam_federated.fl_hiddens,
                                                                    bias_init_prior_prob,
                                                                    linear_regression=linear_regression)
        elif task == UNSUPERVISED:
            weighted_best_model_by_loss = ETM(total_feature,
                                              dedicated_hospital_ids=dedicated_hospital_ids,
                                              run_with_fl=run_with_fl)
        else:
            weighted_best_model_by_loss = LSTM(total_feature)

        weighted_best_model_by_loss = weighted_best_model_by_loss.to(
            self._device)
        weighted_best_model_by_loss.load_state_dict(
            weighted_best_model_weights_by_loss)

        # Save model
        pid = PidUtils.instance().get_pid()
        best_pr_model_path = "{}/PID: {} - weighted_best_pr_model_seed_{}.pt".format(
            self._output_path, pid, seed)
        torch.save(weighted_best_model_weights_by_pr, best_pr_model_path)

        best_loss_model_path = "{}/PID: {} - weighted_best_loss_model_seed_{}.pt".format(
            self._output_path, pid, seed)
        torch.save(weighted_best_model_weights_by_loss, best_loss_model_path)

        # Test
        test_result_by_best_pr_model, test_result_by_best_loss_model = None, None
        if not test_with_bootstrap:
            test_result_by_best_pr_model = self._classifier_service.run_classification(model=weighted_best_model_by_pr,
                                                                                       x=x_target_test, y=y_target_test,
                                                                                       split='test',
                                                                                       linear_regression=linear_regression)

            test_result_by_best_loss_model = self._classifier_service.run_classification(
                model=weighted_best_model_by_loss,
                x=x_target_test, y=y_target_test,
                split='test', linear_regression=linear_regression)

            # LogUtils.instance().log_info("Seed: {} - [Weighted] Target hospital id: {}, test loss by best AUPRC model: {:.5f}, test acc by best AUPRC model: {:.5f}, test roc by best AUPRC model: {:.5f}, test pr by best AUPRC model: {:.5f}, test loss by best loss model: {:.5f}, test acc by best loss model: {:.5f}, test roc by best loss model: {:.5f}, test pr by best loss model: {:.5f}".format(
            #     seed, target_hospital_id, test_result_by_best_pr_model.loss, test_result_by_best_pr_model.acc, test_result_by_best_pr_model.roc, test_result_by_best_pr_model.pr, test_result_by_best_loss_model.loss, test_result_by_best_loss_model.acc, test_result_by_best_loss_model.roc, test_result_by_best_loss_model.pr))

            LogUtils.instance().log_info(
                "Seed: {} - [Weighted] Target hospital id: {}, test loss by best AUPRC model: {:.5f}, test acc by best AUPRC model: {:.5f}, test pr by best AUPRC model: {:.5f}, test loss by best loss model: {:.5f}, test acc by best loss model: {:.5f}, test pr by best loss model: {:.5f}".format(
                    seed, target_hospital_id, test_result_by_best_pr_model.loss, test_result_by_best_pr_model.acc,
                    test_result_by_best_pr_model.pr, test_result_by_best_loss_model.loss,
                    test_result_by_best_loss_model.acc, test_result_by_best_loss_model.pr))

        return train_result_hist, val_result_hist, test_result_by_best_pr_model, test_result_by_best_loss_model, weighted_best_model_by_pr, weighted_best_model_by_loss

    """ Private methods """

    def _train_target_made(self, target_hospital_id, hyperparam_made, x_target_train):

        # Train MADE on target
        # construct model and ship to GPU
        hidden_list = list(map(int, hyperparam_made.made_hiddens.split(',')))
        target_made = MADE(x_target_train.size(1), hidden_list,
                           x_target_train.size(1), num_masks=hyperparam_made.num_masks,
                           natural_ordering=hyperparam_made.natural_ordering)
        target_made.to(self._device)

        # set up the optimizer
        target_made_opt = torch.optim.Adam(target_made.parameters(),
                                           lr=hyperparam_made.made_learning_rate,
                                           weight_decay=hyperparam_made.made_weight_decay)
        target_made_scheduler = torch.optim.lr_scheduler.StepLR(
            target_made_opt, step_size=45, gamma=0.1)

        # start the training
        made_train_hist = []
        epochs_without_improvement = 0
        best_train_loss = float('inf')
        for epoch in range(hyperparam_made.made_epochs):
            # LogUtils.instance().log_info("epoch %d" % (epoch, ))
            # run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
            made_train_loss, _ = self._made_service.run_made(target_made, target_made_opt,
                                                             x_target_train, None, 'train',
                                                             hyperparam_made.batch_size,
                                                             hyperparam_made.samples,
                                                             hyperparam_made.resample_every,
                                                             self._device)

            target_made_scheduler.step()
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
                "Train MADE on target hospital {} - epoch: {}, train loss: {:.5f}".format(target_hospital_id, epoch,
                                                                                          made_train_loss),
                type=LOGGER_MADE)

        # Plot
        pid = PidUtils.instance().get_pid()
        file_name = "{}/PID: {} - target_hospital: {} made_loss.png".format(
            self._output_path, pid, target_hospital_id)

        plt.title("Target hospital {} MADE loss".format(target_hospital_id))
        plt.plot(made_train_hist, label="made_trained_on_target")
        plt.legend()
        plt.grid(0.5)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(file_name)

        return target_made

    def _train_target_vae(self, target_hospital_id, hyperparam_vae, x_target_train):

        # Train VAE on target
        # construct model and ship to GPU
        target_vae = VAE(x_target_train.size(1),
                         hyperparam_vae.vae_hiddens,
                         hyperparam_vae.vae_latent_dim)
        target_vae.to(self._device)

        # set up the optimizer
        target_vae_opt = torch.optim.Adam(target_vae.parameters(),
                                          lr=hyperparam_vae.vae_learning_rate,
                                          weight_decay=hyperparam_vae.vae_weight_decay)
        target_vae_scheduler = torch.optim.lr_scheduler.StepLR(
            target_vae_opt, step_size=45, gamma=0.1)

        # start the training
        vae_train_hist = []
        epochs_without_improvement = 0
        best_train_loss = float('inf')
        for epoch in range(hyperparam_vae.vae_epochs):
            # LogUtils.instance().log_info("epoch %d" % (epoch, ))
            # run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
            beta = 0.001 * epoch
            beta = beta if beta < 1.0 else 1.0
            vae_train_loss, _, recon_loss, kl_loss = self._vae_service.run_vae(target_vae, target_vae_opt,
                                                                               x_target_train, None, 'train',
                                                                               hyperparam_vae.batch_size,
                                                                               beta, self._device)

            target_vae_scheduler.step()
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
                "Train VAE on target hospital {} - epoch: {}, train loss: {:.5f}, recon_loss: {:.5f}, kl_loss: {:.5f}".format(
                    target_hospital_id, epoch, vae_train_loss, recon_loss, kl_loss), type=LOGGER_MADE)

        # Plot
        pid = PidUtils.instance().get_pid()
        file_name = "{}/PID: {} - target_hospital: {} vae_loss.png".format(
            self._output_path, pid, target_hospital_id)

        plt.clf()
        plt.title("Target hospital {} VAE loss".format(target_hospital_id))
        plt.plot(vae_train_hist, label="vae_trained_on_target")

        plt.legend()
        plt.grid(0.5)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(file_name)
        plt.close()

        return target_vae

    def _train_target_vqvae(self, target_hospital_id, hyperparam_vae, x_target_train):

        # Train VQVAE on target
        # construct model and ship to GPU
        target_vqvae = VQVAE(x_target_train.size(1),
                             hyperparam_vae.vae_hiddens,
                             hyperparam_vae.vae_latent_dim,
                             0.25,
                             self._device)
        target_vqvae.to(self._device)

        # set up the optimizer
        target_vqvae_opt = torch.optim.Adam(target_vqvae.parameters(),
                                            lr=hyperparam_vae.vae_learning_rate,
                                            weight_decay=hyperparam_vae.vae_weight_decay)
        target_vqvae_scheduler = torch.optim.lr_scheduler.StepLR(
            target_vqvae_opt, step_size=45, gamma=0.1)

        # start the training
        vqvae_train_hist = []
        epochs_without_improvement = 0
        best_train_loss = float('inf')
        for epoch in range(hyperparam_vae.vae_epochs):
            # LogUtils.instance().log_info("epoch %d" % (epoch, ))
            # run_epoch('test', upto=5) # run only a few batches for approximate test accuracy
            vqvae_train_loss, _ = self._vqvae_service.run_vqvae(target_vqvae, target_vqvae_opt,
                                                                x_target_train, None, 'train',
                                                                hyperparam_vae.batch_size,
                                                                self._device)

            target_vqvae_scheduler.step()
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
                "Train VQVAE on target hospital {} - epoch: {}, train loss: {:.5f}".format(target_hospital_id, epoch,
                                                                                           vqvae_train_loss),
                type=LOGGER_MADE)

        # Plot
        pid = PidUtils.instance().get_pid()
        file_name = "{}/PID: {} - target_hospital: {} vqvae_loss.png".format(
            self._output_path, pid, target_hospital_id)

        plt.clf()
        plt.title("Target hospital {} VQVAE loss".format(target_hospital_id))
        plt.plot(vqvae_train_hist, label="vqvae_trained_on_target")

        plt.legend()
        plt.grid(0.5)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.savefig(file_name)
        plt.close()

        return target_vqvae

    def _train_target_lda(self, target_hospital_id, topics, x_target_train):

        lda_model = LatentDirichletAllocation(n_components=topics,
                                              random_state=42)
        x_target_train = x_target_train.cpu().detach().numpy().copy()
        x_train = csr_matrix(x_target_train)
        lda_model.fit(x_train)

        perplexity = lda_model.perplexity(x_train)

        LogUtils.instance().log_info(
            "Train LDA on target hospital {} perplexity: {:.5f}".format(target_hospital_id, perplexity),
            type=LOGGER_MADE)

        return lda_model
