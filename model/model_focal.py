import torch.nn as nn


class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma):
        super().__init__()
        self._alpha = alpha
        self._gamma = gamma

    def forward(self, inputs, targets):
        criterion = nn.BCELoss(reduction="none")
        ce_loss = criterion(inputs, targets)
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self._gamma)

        if self._alpha is not None and self._alpha >= 0:
            alpha_t = self._alpha * targets + (1 - self._alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss
