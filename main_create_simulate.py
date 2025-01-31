from service.service_simulate import SimulateDataCreateService
from service.data_loader.data_loader_factory import DataLoaderFactory

from utils.utils_pid import PidUtils
from utils.utils_log import LogUtils
from utils.constants import *

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

import numbers
import copy

import hydra
from omegaconf import DictConfig


class SimulatedMLP(nn.Module):

    def __init__(self, in_features):
        super(SimulatedMLP, self).__init__()
        self.classifier = nn.Linear(in_features, 1)
        init_weights = torch.normal(mean=0.0,
                                    std=1.0,
                                    size=self.classifier.weight.shape)
        mask = np.random.binomial(n=1,
                                  p=0.18,
                                  size=self.classifier.weight.shape)
        mask = torch.Tensor(mask)
        init_weights *= mask

        LogUtils.instance().log_info("Init weight: {}".format(init_weights))
        with torch.no_grad():
            self.classifier.weight = torch.nn.Parameter(init_weights)

        bias_init = -np.log((1 - 0.2) / 0.2)
        # bias_init = 0
        LogUtils.instance().log_info("Init bias: {}".format(bias_init))
        nn.init.constant_(self.classifier.bias, bias_init)

    def forward(self, x):
        logits = self.classifier(x)
        return torch.sigmoid(logits)


# class SimulatedMLP(nn.Module):

#     def __init__(self, in_features):
#         super(SimulatedMLP, self).__init__()
#         self.classifier = nn.Linear(in_features, 1)
#         init_weights = torch.normal(mean=0.0,
#                                     std=1.0,
#                                     size=self.classifier.weight.shape)
#         # mask = np.random.binomial(n=1,
#         #                           p=0.15,
#         #                           size=self.classifier.weight.shape)
#         # mask = torch.Tensor(mask)
#         # init_weights *= mask

#         LogUtils.instance().log_info("Init weight: {}".format(init_weights))
#         with torch.no_grad():
#             self.classifier.weight = torch.nn.Parameter(init_weights)

#         # bias_init = -np.log((1 - 0.15) / 0.15)
#         # LogUtils.instance().log_info("Init bias: {}".format(bias_init))
#         # nn.init.constant_(self.classifier.bias, bias_init)
#         nn.init.zeros_(self.classifier.bias)

#     def forward(self, x):
#         logits = self.classifier(x)
#         return torch.sigmoid(logits)


""" Main """


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    pid = PidUtils.instance().get_pid()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    LogUtils.instance().set_log_path(cfg.env.log_path)

    # Load data
    # TODO:
    data_loader_factory = DataLoaderFactory()
    data_loader = data_loader_factory.get_data_loader(VENTILATOR)

    dedicated_hospital_ids = list(
        map(float, cfg.experiment.dedicated_hospital_ids.split(',')))
    if len(dedicated_hospital_ids) < 2:
        raise Exception("Dedicated hospitals shall be more than one")
    hospital_data = data_loader.load_data(dataset_path=cfg.env.dataset_path + "/" + cfg.experiment.eicu_path,
                                          min_death_count=cfg.experiment.min_death_count,
                                          label_idx=LABEL_IDX[VENTILATOR],
                                          hospital_id_col=cfg.experiment.hospital_id_col,
                                          dedicated_hospital_ids=dedicated_hospital_ids)

    hospital_data_list = list(hospital_data.values())

    # Generate simulate data
    mlp = SimulatedMLP(cfg.experiment.total_feature)
    # Debug
    state_dict = torch.load("../data/simulate/41/simulate_mlp_weights.pt")
    weights = state_dict['classifier.weight']
    non_zero_weights = np.where(weights != 0)
    noise = np.random.normal(0, 0.02, non_zero_weights[1].shape).astype(np.float32)
    weights[:, non_zero_weights[1]] += noise
    mlp.load_state_dict(state_dict)
    simulate_data_creator = SimulateDataCreateService(mlp)

    model_weights = mlp.state_dict()

    source_x = []

    for (hospital_id, x, _) in hospital_data_list:

        x = x.astype(np.float16)
        x = torch.Tensor(x).to(device)

        if float(cfg.experiment.target_hospital_id) == hospital_id:

            probs_target, x_target, y_target = simulate_data_creator.generate_fake_data(
                x)

            labels_target, counts_target = np.unique(y_target,
                                                     return_counts=True)
            if counts_target[1] < 350 or counts_target[1] > 450:
                print("Target:")
                print(labels_target)
                print(counts_target)
                raise NotImplementedError()
            
            plt.clf()
            plt.title("Target Output Probabilities")
            plt.hist(probs_target, bins=10)
            plt.savefig(
                "PID: {} - [{}] target_output_probs.png".format(pid, hospital_id))
            plt.close()

            plt.clf()
            plt.title("Target Labels Count")
            plt.bar(labels_target, counts_target, align='center')
            for i, v in enumerate(counts_target):
                plt.text(i-0.1, v, str(v), color='red', fontweight='bold')
            plt.xticks(labels_target)
            plt.savefig(
                "PID: {} - [{}] target_labels_count.png".format(pid, hospital_id))
            plt.close()

            pd.DataFrame(x_target).to_csv(
                "PID: {} - [{}] simulate_x_target.csv".format(pid, hospital_id), index=False)
            pd.DataFrame(y_target).to_csv(
                "PID: {} - [{}] simulate_y_target.csv".format(pid, hospital_id), index=False)
        else:
            source_x.append(x)

    torch.save(model_weights, "PID: {} - simulate_mlp_weights.pt".format(pid))

    source_x = np.vstack(source_x)
    source_x = torch.Tensor(source_x).to(device)
    probs_source, x_source, y_source = simulate_data_creator.generate_fake_data(
        source_x)

    plt.clf()
    plt.title("Source Output Probabilities")
    plt.hist(probs_source, bins=10)
    plt.savefig(
        "PID: {} - source_output_probs.png".format(pid))
    plt.close()

    labels_source, counts_source = np.unique(y_source,
                                             return_counts=True)
    plt.clf()
    plt.title("Source Labels Count")
    plt.bar(labels_source, counts_source, align='center')
    for i, v in enumerate(counts_source):
        plt.text(i-0.1, v, str(v), color='red', fontweight='bold')
    plt.xticks(labels_source)
    plt.savefig(
        "PID: {} - source_labels_count.png".format(pid))
    plt.close()

    pd.DataFrame(x_source).to_csv(
        "PID: {} - simulate_x_source.csv".format(pid), index=False)

    pd.DataFrame(y_source).to_csv(
        "PID: {} - simulate_y_source.csv".format(pid), index=False)


if __name__ == "__main__":
    main()
