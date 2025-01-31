import torch
import copy

from model.result import Result
from model.model_classifier import ModelBinaryClassification
from model.model_lstm import LSTM
from model.model_etm import ETM
from utils.utils_log import LogUtils
from utils.utils_pid import PidUtils
from utils.constants import *


class UnweightedService:
    """ Initialize """

    def __init__(self, federated_service, classifier_service, device, output_path):
        self._federated_service = federated_service
        self._classifier_service = classifier_service
        self._device = device
        self._output_path = output_path

    """ Unweighted """

    def train_unweighted(self, seed, target_hospital_id,
                         source_hospitals,
                         hyperparam_federated,
                         bias_init_prior_prob,
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
            unweighted_global_model = ModelBinaryClassification(total_feature,
                                                                hyperparam_federated.fl_hiddens,
                                                                bias_init_prior_prob,
                                                                linear_regression=linear_regression)
        elif task == UNSUPERVISED:
            unweighted_global_model = ETM(total_feature,
                                          dedicated_hospital_ids=dedicated_hospital_ids,
                                          run_with_fl=run_with_fl)
        else:
            unweighted_global_model = LSTM(total_feature)

        unweighted_global_model = unweighted_global_model.to(self._device)
        unweighted_global_model.load_state_dict(
            copy.deepcopy(global_model_weights))

        unweighted_global_optimizer = torch.optim.Adam(unweighted_global_model.parameters(),
                                                       lr=hyperparam_federated.learning_rate,
                                                       weight_decay=hyperparam_federated.weight_decay)

        # Validation
        unweighted_best_val_pr = float('-inf')
        unweighted_best_val_pr_round = None
        unweighted_best_model_weights_by_pr = None

        unweighted_best_val_loss = float('inf')
        unweighted_best_val_loss_round = None
        unweighted_best_model_weights_by_loss = None

        unweighted_val_loss_hist = []
        unweighted_val_acc_hist = []
        # unweighted_val_roc_hist = []
        unweighted_val_pr_hist = []
        unweighted_val_coherence_hist = []
        unweighted_val_diversity_hist = []
        unweighted_val_quality_hist = []

        val_result_hist.loss_hist = unweighted_val_loss_hist
        val_result_hist.acc_hist = unweighted_val_acc_hist
        # val_result_hist.roc_hist = unweighted_val_roc_hist
        val_result_hist.pr_hist = unweighted_val_pr_hist
        val_result_hist.coherence_hist = unweighted_val_coherence_hist
        val_result_hist.diversity_hist = unweighted_val_diversity_hist
        val_result_hist.quality_hist = unweighted_val_quality_hist

        # Train
        unweighted_train_loss_hist = []
        unweighted_train_acc_hist = []
        # unweighted_train_roc_hist = []
        unweighted_train_pr_hist = []

        train_result_hist.loss_hist = unweighted_train_loss_hist
        train_result_hist.acc_hist = unweighted_train_acc_hist
        # train_result_hist.roc_hist = unweighted_train_roc_hist
        train_result_hist.pr_hist = unweighted_train_pr_hist

        epochs_without_improvement = 0
        for round in range(hyperparam_federated.total_round):

            avg_train, avg_val = self._federated_service.run_fed(round,
                                                                 unweighted_global_model,
                                                                 unweighted_global_optimizer,
                                                                 source_hospitals,
                                                                 x_target_val, y_target_val,
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

            # LogUtils.instance().log_info("Seed: {} - [Unweighted] Target hospital id: {}, round: {}, train: loss: {:.5f}, acc: {:.5f}, roc: {:.5f}, pr: {:.5f}, val: loss: {:.5f}, acc: {:.5f}, roc: {:.5f}, pr: {:.5f}".format(
            #     seed, target_hospital_id, round, train_loss, train_acc, train_roc, train_pr, val_loss, val_acc, val_roc, val_pr))

            LogUtils.instance().log_info(
                "Seed: {} - [Unweighted] Target hospital id: {}, round: {}, train: loss: {:.5f}, acc: {:.5f}, pr: {:.5f}, val: loss: {:.5f}, acc: {:.5f}, pr: {:.5f}".format(
                    seed, target_hospital_id, round, train_loss, train_acc, train_pr, val_loss, val_acc, val_pr))

            unweighted_val_loss_hist.append(val_loss)
            unweighted_val_acc_hist.append(val_acc)
            # unweighted_val_roc_hist.append(val_roc)
            unweighted_val_pr_hist.append(val_pr)

            unweighted_train_loss_hist.append(train_loss)
            unweighted_train_acc_hist.append(train_acc)
            # unweighted_train_roc_hist.append(train_roc)
            unweighted_train_pr_hist.append(train_pr)

            if val_pr > unweighted_best_val_pr:
                unweighted_best_val_pr = val_pr
                unweighted_best_val_pr_round = round
                model_weights = unweighted_global_model.state_dict()
                unweighted_best_model_weights_by_pr = copy.deepcopy(
                    model_weights)
                if not linear_regression and not unsupervised:
                    epochs_without_improvement = 0
            else:
                if not linear_regression and not unsupervised:
                    epochs_without_improvement += 1

            if val_loss < unweighted_best_val_loss:
                unweighted_best_val_loss = val_loss
                unweighted_best_val_loss_round = round
                model_weights = unweighted_global_model.state_dict()
                unweighted_best_model_weights_by_loss = copy.deepcopy(
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
            "Seed: {} - [Unweighted] Target hospital id: {}, Best val pr: {:.5f}, best val pr round: {}, Best val loss: {:.5f}, best val loss round: {}".format(
                seed, target_hospital_id, unweighted_best_val_pr, unweighted_best_val_pr_round,
                unweighted_best_val_loss, unweighted_best_val_loss_round))

        if task == DEATH or task == LENGTH:
            unweighted_best_model_by_pr = ModelBinaryClassification(total_feature,
                                                                    hyperparam_federated.fl_hiddens,
                                                                    bias_init_prior_prob,
                                                                    linear_regression=linear_regression)
        elif task == UNSUPERVISED:
            unweighted_best_model_by_pr = ETM(total_feature,
                                              dedicated_hospital_ids=dedicated_hospital_ids,
                                              run_with_fl=run_with_fl)
        else:
            unweighted_best_model_by_pr = LSTM(total_feature)

        unweighted_best_model_by_pr = unweighted_best_model_by_pr.to(
            self._device)
        unweighted_best_model_by_pr.load_state_dict(
            unweighted_best_model_weights_by_pr)

        if task == DEATH or task == LENGTH:
            unweighted_best_model_by_loss = ModelBinaryClassification(total_feature,
                                                                      hyperparam_federated.fl_hiddens,
                                                                      bias_init_prior_prob,
                                                                      linear_regression=linear_regression)
        elif task == UNSUPERVISED:
            unweighted_best_model_by_loss = ETM(total_feature,
                                                dedicated_hospital_ids=dedicated_hospital_ids,
                                                run_with_fl=run_with_fl)
        else:
            unweighted_best_model_by_loss = LSTM(total_feature)

        unweighted_best_model_by_loss = unweighted_best_model_by_loss.to(
            self._device)
        unweighted_best_model_by_loss.load_state_dict(
            unweighted_best_model_weights_by_loss)

        # Save model
        pid = PidUtils.instance().get_pid()
        best_pr_model_path = "{}/PID: {} - unweighted_best_pr_model_seed_{}.pt".format(
            self._output_path, pid, seed)
        torch.save(unweighted_best_model_weights_by_pr, best_pr_model_path)

        best_loss_model_path = "{}/PID: {} - unweighted_best_loss_model_seed_{}.pt".format(
            self._output_path, pid, seed)
        torch.save(unweighted_best_model_weights_by_loss, best_loss_model_path)

        # Test
        test_result_by_best_pr_model, test_result_by_best_loss_model = None, None
        if not test_with_bootstrap:
            test_result_by_best_pr_model = self._classifier_service.run_classification(
                model=unweighted_best_model_by_pr,
                x=x_target_test, y=y_target_test,
                split='test', linear_regression=linear_regression)

            test_result_by_best_loss_model = self._classifier_service.run_classification(
                model=unweighted_best_model_by_loss,
                x=x_target_test, y=y_target_test,
                split='test', linear_regression=linear_regression)

            # LogUtils.instance().log_info("Seed: {} - [Unweighted] Target hospital id: {}, test loss by best AUPRC model: {:.5f}, test acc by best AUPRC model: {:.5f}, test roc by best AUPRC model: {:.5f}, test pr by best AUPRC model: {:.5f}, test loss by best loss model: {:.5f}, test acc by best loss model: {:.5f}, test roc by best loss model: {:.5f}, test pr by best loss model: {:.5f}".format(
            #     seed, target_hospital_id, test_result_by_best_pr_model.loss, test_result_by_best_pr_model.acc, test_result_by_best_pr_model.roc, test_result_by_best_pr_model.pr, test_result_by_best_loss_model.loss, test_result_by_best_loss_model.acc, test_result_by_best_loss_model.roc, test_result_by_best_loss_model.pr))

            LogUtils.instance().log_info(
                "Seed: {} - [Unweighted] Target hospital id: {}, test loss by best AUPRC model: {:.5f}, test acc by best AUPRC model: {:.5f}, test pr by best AUPRC model: {:.5f}, test loss by best loss model: {:.5f}, test acc by best loss model: {:.5f}, test pr by best loss model: {:.5f}".format(
                    seed, target_hospital_id, test_result_by_best_pr_model.loss, test_result_by_best_pr_model.acc,
                    test_result_by_best_pr_model.pr, test_result_by_best_loss_model.loss,
                    test_result_by_best_loss_model.acc, test_result_by_best_loss_model.pr))

        return train_result_hist, val_result_hist, test_result_by_best_pr_model, test_result_by_best_loss_model, unweighted_best_model_by_pr, unweighted_best_model_by_loss
