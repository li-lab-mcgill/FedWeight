from model.model_classifier import ModelBinaryClassification

import torch

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):

    init_global_model = ModelBinaryClassification(cfg.experiment.total_feature,
                                                  cfg.experiment.fl_hiddens,
                                                  cfg.experiment.bias_init_prior_prob)

    init_model_path = cfg.env.dataset_path + "/" + cfg.experiment.init_global_model
    torch.save(init_global_model.state_dict(), init_model_path)


if __name__ == "__main__":
    main()
