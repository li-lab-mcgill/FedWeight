import torch
import torch.nn as nn
import numpy as np


class ModelBinaryClassification(nn.Module):

    def __init__(self, in_features, hidden_sizes, prior_prob=None, 
                 linear_regression=False):

        super().__init__()

        self._main = []
        self._is_linear = hidden_sizes is None
        self._linear_regression = linear_regression

        if not self._is_linear:

            hidden_list = list(map(int, hidden_sizes.split(',')))
            total_hidden_sizes = [in_features] + hidden_list + [1]
            for h0, h1 in zip(total_hidden_sizes, total_hidden_sizes[1:]):
                self._main.extend([
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ])

            self._main.pop()  # Pop the last ReLU
            self._classifier = self._main.pop()  # Pop the ouput layer

            for layer in self._main:
                if not isinstance(layer, nn.Linear):
                    continue
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

            self._main = nn.Sequential(*self._main)
        else:
            # Linear model
            self._classifier = nn.Linear(in_features, 1)

        bias_init = 0
        # if prior_prob is not None:
        #     bias_init = -np.log((1 - prior_prob) / prior_prob)
        #     print(bias_init)
        #     raise NotImplementedError()

        nn.init.xavier_uniform_(self._classifier.weight)
        nn.init.constant_(self._classifier.bias, bias_init)

    def forward(self, x):
        if not self._is_linear:
            x = self._main(x)
        logits = self._classifier(x)
        if not self._linear_regression:
            return torch.sigmoid(logits)
        else:
            return logits
