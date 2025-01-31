from model.client import Client
from model.model_classifier import ModelBinaryClassification
from model.model_lstm import LSTM
from model.model_etm import ETM
from model.hyperparam_federated import FederatedHyperParam
from model.hyperparam_focal import FocalHyperParam
from model.hyperparam_made import MadeHyperParam
from model.hyperparam_vae import VaeHyperParam
from service.data_loader.data_loader_factory import DataLoaderFactory
from service.service_made import MadeService
from service.service_vae import VaeService
from service.service_vqvae import VqVaeService
from service.classification.classifier_service_factory import ClassifierServiceFactory
from service.service_plot import PlotService
from service.federated.federated_service_factory import FederatedServiceFactory
from service.service_unweighted import UnweightedService
from service.service_weighted import WeightedService
from service.service_data_split import DataSplitService
from utils.utils_log import LogUtils
from utils.utils_pid import PidUtils
from utils.constants import *

import os
import torch
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import scipy.stats as statstat
from scipy.stats import pearsonr

import hydra
from omegaconf import DictConfig, OmegaConf

""" Load data """


def load_federated_data(cfg, data_loader):
    if cfg.experiment.task == SIMULATION:

        hospital_data = data_loader.load_data(
            simulate_x_source_path=cfg.env.dataset_path + "/" + cfg.experiment.simulate_x_source_path,
            simulate_y_source_path=cfg.env.dataset_path +
                                   "/" + cfg.experiment.simulate_y_source_path,
            simulate_x_target_path=cfg.env.dataset_path +
                                   "/" + cfg.experiment.simulate_x_target_path,
            simulate_y_target_path=cfg.env.dataset_path + "/" + cfg.experiment.simulate_y_target_path)

    # elif cfg.experiment.task == COLOR_MNIST or cfg.experiment.task == BINARIZED_MNIST:
    #
    #     hospital_data = data_loader.load_data()

    else:
        if cfg.experiment.dedicated_hospital_ids is not None:
            dedicated_hospital_ids = list(
                map(float, cfg.experiment.dedicated_hospital_ids.split(',')))
            if len(dedicated_hospital_ids) < 2:
                raise Exception("Dedicated hospitals shall be more than one")
            hospital_data = data_loader.load_data(dataset_path=cfg.env.dataset_path + "/" + cfg.experiment.eicu_path,
                                                  time_dataset_path=cfg.env.dataset_path + "/" + cfg.experiment.eicu_time_path,
                                                  min_death_count=cfg.experiment.min_death_count,
                                                  dedicated_hospital_ids=dedicated_hospital_ids,
                                                  task=cfg.experiment.task,
                                                  run_with_fl=cfg.experiment.run_with_fl,
                                                  hospital_id_col=cfg.experiment.hospital_id_col)
        else:
            hospital_data = data_loader.load_data(dataset_path=cfg.env.dataset_path + "/" + cfg.experiment.eicu_path,
                                                  time_dataset_path=cfg.env.dataset_path + "/" + cfg.experiment.eicu_time_path,
                                                  min_death_count=cfg.experiment.min_death_count,
                                                  task=cfg.experiment.task,
                                                  run_with_fl=cfg.experiment.run_with_fl,
                                                  hospital_id_col=cfg.experiment.hospital_id_col)

        if hospital_data is None or len(hospital_data) == 0:
            assert "No data loaded from file path: {}".format(
                cfg.env.dataset_path + "/" + cfg.experiment.eicu_path)

    return list(hospital_data.values())


""" Main """


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    if cfg.env.use_cuda and not torch.cuda.is_available():
        raise Exception("Please use CUDA environment!")

    if not os.path.exists(os.path.join(os.getcwd(), cfg.env.dataset_path)):
        raise FileNotFoundError(
            "Data directory not found. Please create data directory first!")

    if not os.path.exists(os.path.join(os.getcwd(), cfg.env.log_path)):
        raise FileNotFoundError(
            "Log directory not found. Please create log directory first!")

    if not os.path.exists(os.path.join(os.getcwd(), cfg.env.output_path)):
        raise FileNotFoundError(
            "Output directory not found. Please create output directory first!")

    pid = PidUtils.instance().get_pid()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    LogUtils.instance().set_log_path(cfg.env.log_path)

    # Hyperparams
    hyperparam_federated = FederatedHyperParam(cfg.experiment.test_size,
                                               cfg.experiment.val_size,
                                               cfg.experiment.total_feature,
                                               cfg.experiment.fl_hiddens,
                                               cfg.experiment.learning_rate,
                                               cfg.experiment.weight_decay,
                                               cfg.experiment.num_rounds,
                                               cfg.experiment.num_epochs,
                                               cfg.experiment.batch_size)
    hyperparam_made = MadeHyperParam(cfg.experiment.made_epochs,
                                     cfg.experiment.made_hiddens,
                                     cfg.experiment.made_num_masks,
                                     cfg.experiment.made_samples,
                                     cfg.experiment.made_resample_every,
                                     cfg.experiment.made_natural_ordering,
                                     cfg.experiment.made_learning_rate,
                                     cfg.experiment.made_weight_decay,
                                     cfg.experiment.batch_size)
    hyperparam_vae = VaeHyperParam(cfg.experiment.vae_epochs,
                                   cfg.experiment.vae_latent_dim,
                                   cfg.experiment.vae_hiddens,
                                   cfg.experiment.vae_learning_rate,
                                   cfg.experiment.vae_weight_decay,
                                   cfg.experiment.batch_size)
    hyperparam_focal = FocalHyperParam(cfg.experiment.focal_alpha,
                                       cfg.experiment.focal_gamma)

    # Service
    data_loader_factory = DataLoaderFactory()
    data_loader = data_loader_factory.get_data_loader(cfg.experiment.task)

    made_service = MadeService(cfg.experiment.task)
    vae_service = VaeService(cfg.experiment.task)
    vqvae_service = VqVaeService(cfg.experiment.task)
    plot_service = PlotService()
    data_split_service = DataSplitService(device)

    classifier_factory = ClassifierServiceFactory()
    classifier_service = classifier_factory.get_classifier(
        cfg.experiment.fed_weight_method)

    federated_factory = FederatedServiceFactory(classifier_service)
    federated_service = federated_factory.get_service(
        cfg.experiment.fed_weight_method)

    linear_regression = cfg.experiment.task == LENGTH
    unsupervised = cfg.experiment.task == UNSUPERVISED
    unweighted_service = UnweightedService(federated_service,
                                           classifier_service,
                                           device,
                                           cfg.env.output_path)
    weighted_service = WeightedService(federated_service,
                                       classifier_service,
                                       made_service,
                                       vae_service,
                                       vqvae_service,
                                       plot_service,
                                       data_split_service,
                                       device,
                                       cfg.env.output_path,
                                       cfg.experiment.density_estimator)

    # Init all clients
    hospital_data_list = load_federated_data(cfg, data_loader)
    assert hospital_data_list is not None and len(hospital_data_list) > 0

    source_hospitals = []
    cls_x_target, cls_y_target, density_x_target = None, None, None
    cls_x_total, cls_y_total, cls_hospital_total = [], [], []

    if cfg.experiment.run_with_fl:

        for (hospital_id, cls_x, cls_y, density_x) in hospital_data_list:

            LogUtils.instance().log_info(
                "Hospital id: {}, patients: {}, {}, {}".format(hospital_id, cls_x.shape, cls_y.shape, density_x.shape))

            target_indicator = 1.0 if float(cfg.experiment.target_hospital_id) == hospital_id else 0.0
            cls_hospital = np.full(cls_y.shape, target_indicator, dtype=np.float32)
            cls_x_total.append(cls_x)
            cls_y_total.append(cls_y)
            cls_hospital_total.append(cls_hospital)

            if float(cfg.experiment.target_hospital_id) == hospital_id:
                cls_x_target, cls_y_target, density_x_target = cls_x, cls_y, density_x
                continue

            if hospital_id in [-1, -2]:
                continue

            client = Client(hospital_id, cls_x, cls_y, density_x,
                            made_service, vae_service, vqvae_service, classifier_service,
                            data_split_service, plot_service,
                            hyperparam_federated, hyperparam_made, hyperparam_vae, cfg.experiment.lda_topics,
                            hyperparam_focal, device, cfg.experiment.algorithm,
                            cfg.env.output_path, cfg.experiment.task, cfg.experiment.reweight_phi,
                            cfg.experiment.reweight_lambda,
                            cfg.experiment.bias_init_prior_prob,
                            linear_regression=linear_regression,
                            unsupervised=unsupervised,
                            test_with_bootstrap=cfg.experiment.test_with_bootstrap,
                            density_estimator=cfg.experiment.density_estimator,
                            dedicated_hospital_ids=cfg.experiment.dedicated_hospital_ids,
                            run_with_fl=cfg.experiment.run_with_fl)
            source_hospitals.append(client)
    else:

        cls_x_list, cls_y_list, density_x_list = [], [], []
        for (hospital_id, cls_x, cls_y, density_x) in hospital_data_list:

            LogUtils.instance().log_info(
                "Hospital id: {}, patients: {}, {}, {}".format(hospital_id, cls_x.shape, cls_y.shape, density_x.shape))

            target_indicator = 1.0 if float(cfg.experiment.target_hospital_id) == hospital_id else 0.0
            cls_hospital = np.full(cls_y.shape, target_indicator, dtype=np.float32)
            cls_x_total.append(cls_x)
            cls_y_total.append(cls_y)
            cls_hospital_total.append(cls_hospital)

            if float(cfg.experiment.target_hospital_id) == hospital_id:
                cls_x_target, cls_y_target, density_x_target = cls_x, cls_y, density_x
            else:
                if hospital_id not in [-1, -2]:
                    cls_x_list.append(cls_x)
                    cls_y_list.append(cls_y)
                    density_x_list.append(density_x)

        cls_x_source = np.concatenate(cls_x_list, axis=0)
        cls_y_source = np.concatenate(cls_y_list, axis=0)
        density_x_source = np.concatenate(density_x_list, axis=0)

        client = Client(hospital_id, cls_x_source, cls_y_source, density_x_source,
                        made_service, vae_service, vqvae_service, classifier_service,
                        data_split_service, plot_service,
                        hyperparam_federated, hyperparam_made, hyperparam_vae, cfg.experiment.lda_topics,
                        hyperparam_focal, device, cfg.experiment.algorithm,
                        cfg.env.output_path, cfg.experiment.task, cfg.experiment.reweight_phi,
                        cfg.experiment.reweight_lambda,
                        cfg.experiment.bias_init_prior_prob,
                        linear_regression=linear_regression,
                        unsupervised=unsupervised,
                        test_with_bootstrap=cfg.experiment.test_with_bootstrap,
                        density_estimator=cfg.experiment.density_estimator,
                        dedicated_hospital_ids=cfg.experiment.dedicated_hospital_ids,
                        run_with_fl=cfg.experiment.run_with_fl)
        source_hospitals.append(client)

    assert cls_x_target is not None
    assert cls_y_target is not None
    assert density_x_target is not None
    assert len(source_hospitals) > 0

    cls_x_total = np.concatenate(cls_x_total, axis=0)
    cls_y_total = np.concatenate(cls_y_total, axis=0)
    cls_hospital_total = np.concatenate(cls_hospital_total, axis=0)

    cls_x_total = cls_x_total.astype(np.float32)
    cls_y_total = cls_y_total.astype(np.float32)
    cls_hospital_total = cls_hospital_total.astype(np.float32)

    unsupervised_x = torch.from_numpy(cls_x_total).float().to(device)

    total_seed = cfg.experiment.total_seed if not cfg.experiment.test_with_bootstrap else 1

    weighted_train_loss_hist, weighted_train_pr_hist, weighted_val_loss_hist, weighted_val_pr_hist = np.zeros(
        (total_seed), dtype=object), np.zeros(
        (total_seed), dtype=object), np.zeros(
        (total_seed), dtype=object), np.zeros(
        (total_seed), dtype=object)

    unweighted_train_loss_hist, unweighted_train_pr_hist, unweighted_val_loss_hist, unweighted_val_pr_hist = np.zeros(
        (total_seed), dtype=object), np.zeros(
        (total_seed), dtype=object), np.zeros(
        (total_seed), dtype=object), np.zeros(
        (total_seed), dtype=object)

    weighted_test_loss_hist_by_pr, weighted_test_roc_hist_by_pr, weighted_test_pr_hist_by_pr, weighted_test_loss_hist_by_loss, weighted_test_roc_hist_by_loss, weighted_test_pr_hist_by_loss, weighted_test_f1_hist_by_pr, weighted_test_f1_hist_by_loss, weighted_test_ari_hist_by_loss = np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object)

    unweighted_test_loss_hist_by_pr, unweighted_test_roc_hist_by_pr, unweighted_test_pr_hist_by_pr, unweighted_test_loss_hist_by_loss, unweighted_test_roc_hist_by_loss, unweighted_test_pr_hist_by_loss, unweighted_test_f1_hist_by_pr, unweighted_test_f1_hist_by_loss, unweighted_test_ari_hist_by_loss = np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object), np.zeros(
        (cfg.experiment.total_seed), dtype=object)

    # Target hospital
    cls_x_target = np.asarray(cls_x_target)
    cls_x_target = cls_x_target.astype(np.float32)

    cls_y_target = np.asarray(cls_y_target)
    cls_y_target = cls_y_target.astype(np.float32)
    # cls_y_target = cls_y_target.reshape(-1, 1)

    density_x_target = np.asarray(density_x_target)
    density_x_target = density_x_target.astype(np.float32)

    if cfg.experiment.test_with_bootstrap:
        # Bootstrap test set from x_target_non_train and y_target_non_train
        x_target_val, x_target_non_train, y_target_val, y_target_non_train = train_test_split(
            cls_x_target, cls_y_target,
            test_size=cfg.experiment.val_size,
            random_state=42)

    # Federated
    for idx in range(total_seed):

        # Reinitialize data splitting and retrain made
        for client in source_hospitals:
            if cfg.experiment.algorithm != UNWEIGHTED:
                client.train_density_estimator()

        # Reinitialize model parameters
        total_feature = hyperparam_federated.total_feature
        if cfg.experiment.task == DEATH or cfg.experiment.task == LENGTH:
            init_global_model = ModelBinaryClassification(total_feature,
                                                          hyperparam_federated.fl_hiddens,
                                                          cfg.experiment.bias_init_prior_prob,
                                                          linear_regression=linear_regression)
        elif cfg.experiment.task == UNSUPERVISED:
            init_global_model = ETM(total_feature,
                                    dedicated_hospital_ids=cfg.experiment.dedicated_hospital_ids,
                                    run_with_fl=cfg.experiment.run_with_fl)
        else:
            init_global_model = LSTM(total_feature)

        init_global_model = init_global_model.to(device)

        if cfg.experiment.init_global_model is not None:
            init_model_path = cfg.env.dataset_path + "/" + cfg.experiment.init_global_model
            init_global_model.load_state_dict(torch.load(init_model_path))

        global_model_weights = copy.deepcopy(init_global_model.state_dict())

        if cfg.experiment.test_with_bootstrap:

            x_target_test, y_target_test = None, None  # Bootstrap test after training

            x_target_train = torch.Tensor(density_x_target).to(device)

            x_target_val = torch.Tensor(x_target_val).to(device)
            y_target_val = torch.Tensor(y_target_val).to(device)
        else:

            x_target_val, x_target_test, y_target_val, y_target_test = train_test_split(cls_x_target, cls_y_target,
                                                                                        test_size=cfg.experiment.val_size,
                                                                                        random_state=42)

            x_target_train = torch.Tensor(density_x_target).to(device)

            x_target_val = torch.Tensor(x_target_val).to(device)
            y_target_val = torch.Tensor(y_target_val).to(device)

            x_target_test = torch.Tensor(x_target_test).to(device)
            y_target_test = torch.Tensor(y_target_test).to(device)

        # Unweighted
        unweighted_best_model_by_pr, unweighted_best_model_by_loss = None, None
        if cfg.experiment.algorithm == BOTH or cfg.experiment.algorithm == UNWEIGHTED:
            unweighted_train_result_hist, unweighted_val_result_hist, unweighted_test_result_by_pr, unweighted_test_result_by_loss, unweighted_best_model_by_pr, unweighted_best_model_by_loss = unweighted_service.train_unweighted(
                idx,
                cfg.experiment.target_hospital_id,
                source_hospitals,
                hyperparam_federated,
                cfg.experiment.bias_init_prior_prob,
                x_target_val, y_target_val,
                x_target_test, y_target_test,
                global_model_weights,
                total_feature,
                cfg.experiment.task,
                cfg.experiment.test_with_bootstrap,
                linear_regression=linear_regression,
                dedicated_hospital_ids=cfg.experiment.dedicated_hospital_ids,
                run_with_fl=cfg.experiment.run_with_fl,
                unsupervised=unsupervised)
            unweighted_train_loss_hist[idx] = unweighted_train_result_hist.loss_hist
            unweighted_train_pr_hist[idx] = unweighted_train_result_hist.pr_hist
            unweighted_val_loss_hist[idx] = unweighted_val_result_hist.loss_hist
            unweighted_val_pr_hist[idx] = unweighted_val_result_hist.pr_hist

            if not cfg.experiment.test_with_bootstrap:
                unweighted_test_loss_hist_by_pr[idx] = unweighted_test_result_by_pr.loss
                # unweighted_test_roc_hist_by_pr[idx] = unweighted_test_result_by_pr.roc
                unweighted_test_pr_hist_by_pr[idx] = unweighted_test_result_by_pr.pr
                unweighted_test_f1_hist_by_pr[idx] = unweighted_test_result_by_pr.f1_score
                unweighted_test_loss_hist_by_loss[idx] = unweighted_test_result_by_loss.loss
                # unweighted_test_roc_hist_by_loss[idx] = unweighted_test_result_by_loss.roc
                unweighted_test_pr_hist_by_loss[idx] = unweighted_test_result_by_loss.pr
                unweighted_test_f1_hist_by_loss[idx] = unweighted_test_result_by_loss.f1_score

        # Weighted
        weighted_best_model_by_pr, weighted_best_model_by_loss = None, None
        if cfg.experiment.algorithm == BOTH or cfg.experiment.algorithm == WEIGHTED:
            weighted_train_result_hist, weighted_val_result_hist, weighted_test_result_by_pr, weighted_test_result_by_loss, weighted_best_model_by_pr, weighted_best_model_by_loss = weighted_service.train_weighted(
                idx,
                cfg.experiment.target_hospital_id,
                source_hospitals,
                hyperparam_federated,
                hyperparam_made,
                hyperparam_vae,
                cfg.experiment.lda_topics,
                cfg.experiment.bias_init_prior_prob,
                x_target_train,
                x_target_val, y_target_val,
                x_target_test, y_target_test,
                global_model_weights,
                total_feature,
                cfg.experiment.task,
                cfg.experiment.test_with_bootstrap,
                linear_regression=linear_regression,
                dedicated_hospital_ids=cfg.experiment.dedicated_hospital_ids,
                run_with_fl=cfg.experiment.run_with_fl,
                unsupervised=unsupervised)
            weighted_train_loss_hist[idx] = weighted_train_result_hist.loss_hist
            weighted_train_pr_hist[idx] = weighted_train_result_hist.pr_hist
            weighted_val_loss_hist[idx] = weighted_val_result_hist.loss_hist
            weighted_val_pr_hist[idx] = weighted_val_result_hist.pr_hist

            if not cfg.experiment.test_with_bootstrap:
                weighted_test_loss_hist_by_pr[idx] = weighted_test_result_by_pr.loss
                # weighted_test_roc_hist_by_pr[idx] = weighted_test_result_by_pr.roc
                weighted_test_pr_hist_by_pr[idx] = weighted_test_result_by_pr.pr
                weighted_test_f1_hist_by_pr[idx] = weighted_test_result_by_pr.f1_score
                weighted_test_loss_hist_by_loss[idx] = weighted_test_result_by_loss.loss
                # weighted_test_roc_hist_by_loss[idx] = weighted_test_result_by_loss.roc
                weighted_test_pr_hist_by_loss[idx] = weighted_test_result_by_loss.pr
                weighted_test_f1_hist_by_loss[idx] = weighted_test_result_by_loss.f1_score

    """ Experiment Result """

    if cfg.experiment.test_with_bootstrap:
        for idx in range(cfg.experiment.total_seed):

            if unsupervised:
                x_target_test, y_target_test, x_target_binary = resample(cls_x_total, cls_y_total, cls_hospital_total,
                                                                         replace=True)
                # x_target_test, y_target_test, x_target_binary = cls_x_total, cls_y_total, cls_hospital_total
            else:
                x_target_test, y_target_test = resample(x_target_non_train,
                                                        y_target_non_train,
                                                        replace=True)
                x_target_binary = None

            x_target_test = torch.Tensor(x_target_test).to(device)
            y_target_test = torch.Tensor(y_target_test).to(device)

            # Unweighted
            if cfg.experiment.algorithm == BOTH or cfg.experiment.algorithm == UNWEIGHTED:
                unweighted_test_result_by_pr = classifier_service.run_classification(model=unweighted_best_model_by_pr,
                                                                                     x=x_target_test, y=y_target_test,
                                                                                     split='test',
                                                                                     unsupervised_x=unsupervised_x,
                                                                                     linear_regression=linear_regression,
                                                                                     unsupervised=unsupervised)
                unweighted_test_result_by_loss = classifier_service.run_classification(
                    model=unweighted_best_model_by_loss,
                    x=x_target_test, y=y_target_test,
                    target_binary=x_target_binary,
                    unsupervised_x=unsupervised_x,
                    split='test',
                    linear_regression=linear_regression,
                    unsupervised=unsupervised,
                    PID=pid,
                    need_plot=(idx == 0))
                unweighted_test_loss_hist_by_pr[idx] = unweighted_test_result_by_pr.loss
                # unweighted_test_roc_hist_by_pr[idx] = unweighted_test_result_by_pr.roc
                unweighted_test_pr_hist_by_pr[idx] = unweighted_test_result_by_pr.pr
                unweighted_test_f1_hist_by_pr[idx] = unweighted_test_result_by_pr.f1_score
                unweighted_test_loss_hist_by_loss[idx] = unweighted_test_result_by_loss.loss
                # unweighted_test_roc_hist_by_loss[idx] = unweighted_test_result_by_loss.roc
                unweighted_test_pr_hist_by_loss[idx] = unweighted_test_result_by_loss.pr
                unweighted_test_f1_hist_by_loss[idx] = unweighted_test_result_by_loss.f1_score
                unweighted_test_ari_hist_by_loss[idx] = unweighted_test_result_by_loss.ari
                # unweighted_test_coherence_hist_by_loss[idx] = unweighted_test_result_by_loss.coherence
                # unweighted_test_diversity_hist_by_loss[idx] = unweighted_test_result_by_loss.diversity
                # unweighted_test_quality_hist_by_loss[idx] = unweighted_test_result_by_loss.quality

            # Weighted
            if cfg.experiment.algorithm == BOTH or cfg.experiment.algorithm == WEIGHTED:
                weighted_test_result_by_pr = classifier_service.run_classification(model=weighted_best_model_by_pr,
                                                                                   x=x_target_test, y=y_target_test,
                                                                                   split='test',
                                                                                   linear_regression=linear_regression,
                                                                                   unsupervised=unsupervised)
                weighted_test_result_by_loss = classifier_service.run_classification(model=weighted_best_model_by_loss,
                                                                                     x=x_target_test, y=y_target_test,
                                                                                     split='test',
                                                                                     unsupervised_x=unsupervised_x,
                                                                                     target_binary=x_target_binary,
                                                                                     linear_regression=linear_regression,
                                                                                     unsupervised=unsupervised,
                                                                                     PID=pid,
                                                                                     need_plot=(idx == 0))
                weighted_test_loss_hist_by_pr[idx] = weighted_test_result_by_pr.loss
                # weighted_test_roc_hist_by_pr[idx] = weighted_test_result_by_pr.roc
                weighted_test_pr_hist_by_pr[idx] = weighted_test_result_by_pr.pr
                weighted_test_f1_hist_by_pr[idx] = weighted_test_result_by_pr.f1_score
                weighted_test_loss_hist_by_loss[idx] = weighted_test_result_by_loss.loss
                # weighted_test_roc_hist_by_loss[idx] = weighted_test_result_by_loss.roc
                weighted_test_pr_hist_by_loss[idx] = weighted_test_result_by_loss.pr
                weighted_test_f1_hist_by_loss[idx] = weighted_test_result_by_loss.f1_score
                weighted_test_ari_hist_by_loss[idx] = weighted_test_result_by_loss.ari
                # weighted_test_coherence_hist_by_loss[idx] = weighted_test_result_by_loss.coherence
                # weighted_test_diversity_hist_by_loss[idx] = weighted_test_result_by_loss.diversity
                # weighted_test_quality_hist_by_loss[idx] = weighted_test_result_by_loss.quality

    # Unweighted
    if cfg.experiment.algorithm == BOTH or cfg.experiment.algorithm == UNWEIGHTED:
        test_unweighted_loss_avg_by_pr, test_unweighted_loss_std_by_pr = np.mean(
            unweighted_test_loss_hist_by_pr), np.std(unweighted_test_loss_hist_by_pr)
        # test_unweighted_roc_avg_by_pr, test_unweighted_roc_std_by_pr = np.mean(
        #     unweighted_test_roc_hist_by_pr), np.std(unweighted_test_roc_hist_by_pr)
        test_unweighted_pr_avg_by_pr, test_unweighted_pr_std_by_pr = np.mean(
            unweighted_test_pr_hist_by_pr), np.std(unweighted_test_pr_hist_by_pr)
        print(unweighted_test_pr_hist_by_pr)
        test_unweighted_f1_avg_by_pr, test_unweighted_f1_std_by_pr = np.mean(
            unweighted_test_f1_hist_by_pr), np.std(unweighted_test_f1_hist_by_pr)

        test_unweighted_loss_avg_by_loss, test_unweighted_loss_std_by_loss = np.mean(
            unweighted_test_loss_hist_by_loss), np.std(unweighted_test_loss_hist_by_loss)
        # test_unweighted_roc_avg_by_loss, test_unweighted_roc_std_by_loss = np.mean(
        #     unweighted_test_roc_hist_by_loss), np.std(unweighted_test_roc_hist_by_loss)
        test_unweighted_pr_avg_by_loss, test_unweighted_pr_std_by_loss = np.mean(
            unweighted_test_pr_hist_by_loss), np.std(unweighted_test_pr_hist_by_loss)
        test_unweighted_f1_avg_by_loss, test_unweighted_f1_std_by_loss = np.mean(
            unweighted_test_f1_hist_by_loss), np.std(unweighted_test_f1_hist_by_loss)
        test_unweighted_ari_avg_by_loss, test_unweighted_ari_std_by_loss = np.mean(
            unweighted_test_ari_hist_by_loss), np.std(unweighted_test_ari_hist_by_loss)
        # test_unweighted_coherence_avg_by_loss, test_unweighted_coherence_std_by_loss = np.mean(
        #     unweighted_test_coherence_hist_by_loss), np.std(unweighted_test_coherence_hist_by_loss)
        # test_unweighted_diversity_avg_by_loss, test_unweighted_diversity_std_by_loss = np.mean(
        #     unweighted_test_diversity_hist_by_loss), np.std(unweighted_test_diversity_hist_by_loss)
        # test_unweighted_quality_avg_by_loss, test_unweighted_quality_std_by_loss = np.mean(
        #     unweighted_test_quality_hist_by_loss), np.std(unweighted_test_quality_hist_by_loss)

        unweighted_log_msg = f"{OmegaConf.to_yaml(cfg)} - "
        unweighted_log_msg += f"Unweighted test loss by best AUPRC model: {test_unweighted_loss_avg_by_pr:.5f}±{test_unweighted_loss_std_by_pr:.5f}, "
        # unweighted_log_msg += f"test AUROC by best AUPRC model: {test_unweighted_roc_avg_by_pr:.5f}±{test_unweighted_roc_std_by_pr:.5f}, "
        unweighted_log_msg += f"test AUPRC by best AUPRC model: {test_unweighted_pr_avg_by_pr:.5f}±{test_unweighted_pr_std_by_pr:.5f}, "
        unweighted_log_msg += f"test f1 score by best AUPRC model: {test_unweighted_f1_avg_by_pr:.5f}±{test_unweighted_f1_std_by_pr:.5f}, "
        unweighted_log_msg += f"test loss by best loss model: {test_unweighted_loss_avg_by_loss:.5f}±{test_unweighted_loss_std_by_loss:.5f}, "
        # unweighted_log_msg += f"test AUROC by best loss model: {test_unweighted_roc_avg_by_loss:.5f}±{test_unweighted_roc_std_by_loss:.5f}, "
        unweighted_log_msg += f"test AUPRC by best loss model: {test_unweighted_pr_avg_by_loss:.5f}±{test_unweighted_pr_std_by_loss:.5f}, "
        unweighted_log_msg += f"test f1 score by best loss model: {test_unweighted_f1_avg_by_loss:.5f}±{test_unweighted_f1_std_by_loss:.5f}, "
        unweighted_log_msg += f"test ARI score by best loss model: {test_unweighted_ari_avg_by_loss:.5f}±{test_unweighted_ari_std_by_loss:.5f}"
        # unweighted_log_msg += f"test topic coherence by best loss model: {test_unweighted_coherence_avg_by_loss:.5f}±{test_unweighted_coherence_std_by_loss:.5f}, "
        # unweighted_log_msg += f"test topic diversity by best loss model: {test_unweighted_diversity_avg_by_loss:.5f}±{test_unweighted_diversity_std_by_loss:.5f}, "
        # unweighted_log_msg += f"test topic quality by best loss model: {test_unweighted_quality_avg_by_loss:.5f}±{test_unweighted_quality_std_by_loss:.5f}"

        LogUtils.instance().log_info(unweighted_log_msg)
        LogUtils.instance().log_info("Unweighted unweighted_test_loss_hist_by_pr:{}".format(
            unweighted_test_loss_hist_by_pr.tolist()))
        # LogUtils.instance().log_info("Unweighted unweighted_test_roc_hist_by_pr:{}".format(
        #     unweighted_test_roc_hist_by_pr.tolist()))
        LogUtils.instance().log_info(
            "Unweighted unweighted_test_pr_hist_by_pr:{}".format(unweighted_test_pr_hist_by_pr.tolist()))
        LogUtils.instance().log_info(
            "Unweighted unweighted_test_f1_hist_by_pr:{}".format(unweighted_test_f1_hist_by_pr.tolist()))

        LogUtils.instance().log_info("Unweighted unweighted_test_loss_hist_by_loss:{}".format(
            unweighted_test_loss_hist_by_loss.tolist()))
        # LogUtils.instance().log_info("Unweighted unweighted_test_roc_hist_by_loss:{}".format(
        #     unweighted_test_roc_hist_by_loss.tolist()))
        LogUtils.instance().log_info("Unweighted unweighted_test_pr_hist_by_loss:{}".format(
            unweighted_test_pr_hist_by_loss.tolist()))
        LogUtils.instance().log_info("Unweighted unweighted_test_f1_hist_by_loss:{}".format(
            unweighted_test_f1_hist_by_loss.tolist()))
        LogUtils.instance().log_info("Unweighted unweighted_test_ari_hist_by_loss:{}".format(
            unweighted_test_ari_hist_by_loss.tolist()))
        # LogUtils.instance().log_info("Unweighted unweighted_test_coherence_hist_by_loss:{}".format(
        #     unweighted_test_coherence_hist_by_loss.tolist()))
        # LogUtils.instance().log_info("Unweighted unweighted_test_diversity_hist_by_loss:{}".format(
        #     unweighted_test_diversity_hist_by_loss.tolist()))
        # LogUtils.instance().log_info("Unweighted unweighted_test_quality_hist_by_loss:{}".format(
        #     unweighted_test_quality_hist_by_loss.tolist()))

    # Weighted
    if cfg.experiment.algorithm == BOTH or cfg.experiment.algorithm == WEIGHTED:
        test_weighted_loss_avg_by_pr, test_weighted_loss_std_by_pr = np.mean(
            weighted_test_loss_hist_by_pr), np.std(weighted_test_loss_hist_by_pr)
        # test_weighted_roc_avg_by_pr, test_weighted_roc_std_by_pr = np.mean(
        #     weighted_test_roc_hist_by_pr), np.std(weighted_test_roc_hist_by_pr)
        test_weighted_pr_avg_by_pr, test_weighted_pr_std_by_pr = np.mean(
            weighted_test_pr_hist_by_pr), np.std(weighted_test_pr_hist_by_pr)
        test_weighted_f1_avg_by_pr, test_weighted_f1_std_by_pr = np.mean(
            weighted_test_f1_hist_by_pr), np.std(weighted_test_f1_hist_by_pr)

        test_weighted_loss_avg_by_loss, test_weighted_loss_std_by_loss = np.mean(
            weighted_test_loss_hist_by_loss), np.std(weighted_test_loss_hist_by_loss)
        # test_weighted_roc_avg_by_loss, test_weighted_roc_std_by_loss = np.mean(
        #     weighted_test_roc_hist_by_loss), np.std(weighted_test_roc_hist_by_loss)
        test_weighted_pr_avg_by_loss, test_weighted_pr_std_by_loss = np.mean(
            weighted_test_pr_hist_by_loss), np.std(weighted_test_pr_hist_by_loss)
        test_weighted_f1_avg_by_loss, test_weighted_f1_std_by_loss = np.mean(
            weighted_test_f1_hist_by_loss), np.std(weighted_test_f1_hist_by_loss)
        test_weighted_ari_avg_by_loss, test_weighted_ari_std_by_loss = np.mean(
            weighted_test_ari_hist_by_loss), np.std(weighted_test_ari_hist_by_loss)
        # test_weighted_coherence_avg_by_loss, test_weighted_coherence_std_by_loss = np.mean(
        #     weighted_test_coherence_hist_by_loss), np.std(weighted_test_coherence_hist_by_loss)
        # test_weighted_diversity_avg_by_loss, test_weighted_diversity_std_by_loss = np.mean(
        #     weighted_test_diversity_hist_by_loss), np.std(weighted_test_diversity_hist_by_loss)
        # test_weighted_quality_avg_by_loss, test_weighted_quality_std_by_loss = np.mean(
        #     weighted_test_quality_hist_by_loss), np.std(weighted_test_quality_hist_by_loss)

        weighted_log_msg = f"{OmegaConf.to_yaml(cfg)} - "
        weighted_log_msg += f"Weighted test loss by best AUPRC model: {test_weighted_loss_avg_by_pr:.5f}±{test_weighted_loss_std_by_pr:.5f}, "
        # weighted_log_msg += f"test AUROC by best AUPRC model: {test_weighted_roc_avg_by_pr:.5f}±{test_weighted_roc_std_by_pr:.5f}, "
        weighted_log_msg += f"test AUPRC by best AUPRC model: {test_weighted_pr_avg_by_pr:.5f}±{test_weighted_pr_std_by_pr:.5f}, "
        weighted_log_msg += f"test f1 score by best AUPRC model: {test_weighted_f1_avg_by_pr:.5f}±{test_weighted_f1_std_by_pr:.5f}, "
        weighted_log_msg += f"test loss by best loss model: {test_weighted_loss_avg_by_loss:.5f}±{test_weighted_loss_std_by_loss:.5f}, "
        # weighted_log_msg += f"test AUROC by best loss model: {test_weighted_roc_avg_by_loss:.5f}±{test_weighted_roc_std_by_loss:.5f}, "
        weighted_log_msg += f"test AUPRC by best loss model: {test_weighted_pr_avg_by_loss:.5f}±{test_weighted_pr_std_by_loss:.5f}, "
        weighted_log_msg += f"test f1 score by best loss model: {test_weighted_f1_avg_by_loss:.5f}±{test_weighted_f1_std_by_loss:.5f}, "
        weighted_log_msg += f"test ARI score by best loss model: {test_weighted_ari_avg_by_loss:.5f}±{test_weighted_ari_std_by_loss:.5f}"
        # weighted_log_msg += f"test topic coherence by best loss model: {test_weighted_coherence_avg_by_loss:.5f}±{test_weighted_coherence_std_by_loss:.5f}, "
        # weighted_log_msg += f"test topic diversity by best loss model: {test_weighted_diversity_avg_by_loss:.5f}±{test_weighted_diversity_std_by_loss:.5f}, "
        # weighted_log_msg += f"test topic quality by best loss model: {test_weighted_quality_avg_by_loss:.5f}±{test_weighted_quality_std_by_loss:.5f}"

        LogUtils.instance().log_info(weighted_log_msg)
        LogUtils.instance().log_info("Weighted weighted_test_loss_hist_by_pr:{}".format(
            weighted_test_loss_hist_by_pr.tolist()))
        # LogUtils.instance().log_info("Weighted weighted_test_roc_hist_by_pr:{}".format(
        #     weighted_test_roc_hist_by_pr.tolist()))
        LogUtils.instance().log_info(
            "Weighted weighted_test_pr_hist_by_pr:{}".format(weighted_test_pr_hist_by_pr.tolist()))
        LogUtils.instance().log_info(
            "Weighted weighted_test_f1_hist_by_pr:{}".format(weighted_test_f1_hist_by_pr.tolist()))

        LogUtils.instance().log_info("Weighted weighted_test_loss_hist_by_loss:{}".format(
            weighted_test_loss_hist_by_loss.tolist()))
        # LogUtils.instance().log_info("Weighted weighted_test_roc_hist_by_loss:{}".format(
        #     weighted_test_roc_hist_by_loss.tolist()))
        LogUtils.instance().log_info("Weighted weighted_test_pr_hist_by_loss:{}".format(
            weighted_test_pr_hist_by_loss.tolist()))
        LogUtils.instance().log_info("Weighted weighted_test_f1_hist_by_loss:{}".format(
            weighted_test_f1_hist_by_loss.tolist()))
        LogUtils.instance().log_info("Weighted weighted_test_ari_hist_by_loss:{}".format(
            weighted_test_ari_hist_by_loss.tolist()))
        # LogUtils.instance().log_info("Weighted weighted_test_coherence_hist_by_loss:{}".format(
        #     weighted_test_coherence_hist_by_loss.tolist()))
        # LogUtils.instance().log_info("Weighted weighted_test_diversity_hist_by_loss:{}".format(
        #     weighted_test_diversity_hist_by_loss.tolist()))
        # LogUtils.instance().log_info("Weighted weighted_test_quality_hist_by_loss:{}".format(
        #     weighted_test_quality_hist_by_loss.tolist()))

    if cfg.experiment.algorithm == BOTH and not cfg.experiment.test_with_bootstrap:
        paired_loss = stats.ttest_rel(
            weighted_test_loss_hist_by_loss, unweighted_test_loss_hist_by_loss)
        paired_pr = stats.ttest_rel(
            weighted_test_pr_hist_by_pr, unweighted_test_pr_hist_by_pr)
        LogUtils.instance().log_info("Paired T-test for Loss: {}".format(paired_loss))
        LogUtils.instance().log_info("Paired T-test for PR: {}".format(paired_pr))

    # Analyze causal
    if cfg.experiment.task == SIMULATION:

        original_path = cfg.env.dataset_path + "/" + \
                        cfg.experiment.simulate_original_path
        original = torch.load(original_path)
        original_weights = original['classifier.weight'][0].cpu(
        ).detach().numpy().copy()

        # True causal
        original_causal = np.where(
            original_weights != 0, 1, original_weights).astype(int)

        LogUtils.instance().log_info(
            "Original model weights: {}".format(original_weights.tolist()))
        LogUtils.instance().log_info(
            "Original model causal: {}".format(original_causal.tolist()))

        if unweighted_best_model_by_pr is not None:
            unweighted_weights = unweighted_best_model_by_pr._classifier.weight[0].cpu(
            ).detach().numpy().copy()
            LogUtils.instance().log_info(
                "Unweighted model weights: {}".format(unweighted_weights.tolist()))
            LogUtils.instance().log_info("Unweighted model bias: {}".format(
                unweighted_best_model_by_pr._classifier.bias[0].cpu().detach().numpy().copy()))
            unweighted_coefficient, _ = pearsonr(
                original_weights, unweighted_weights)
            LogUtils.instance().log_info(
                "Unweighted model coefficient: {}".format(unweighted_coefficient))

            unweighted_prob = np.abs(unweighted_weights)
            unweighted_precision, unweighted_recall, _ = precision_recall_curve(
                original_causal, unweighted_prob)
            unweighted_auc = auc(unweighted_recall, unweighted_precision)
            LogUtils.instance().log_info("Unweighted model AUPRC: {}".format(unweighted_auc))

        if weighted_best_model_by_pr is not None:
            weighted_weights = weighted_best_model_by_pr._classifier.weight[0].cpu(
            ).detach().numpy().copy()
            LogUtils.instance().log_info(
                "Weighted model weights: {}".format(weighted_weights.tolist()))
            LogUtils.instance().log_info("Weighted model bias: {}".format(
                weighted_best_model_by_pr._classifier.bias[0].cpu().detach().numpy().copy()))
            weighted_coefficient, _ = pearsonr(
                original_weights, weighted_weights)
            LogUtils.instance().log_info(
                "Weighted model coefficient: {}".format(weighted_coefficient))

            weighted_prob = np.abs(weighted_weights)
            weighted_precision, weighted_recall, _ = precision_recall_curve(
                original_causal, weighted_prob)
            weighted_auc = auc(weighted_recall, weighted_precision)
            LogUtils.instance().log_info("Weighted model AUPRC: {}".format(weighted_auc))


if __name__ == "__main__":
    main()
