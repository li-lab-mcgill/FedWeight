import torch
from torch.autograd import Variable
import math

from utils.utils_eval import EvaluationUtils
from utils.constants import *


class VqVaeService:

    def __init__(self, task: str) -> None:
        super(VqVaeService, self).__init__()
        self._task = task

    def run_vqvae(self,
                  model, opt,
                  x_train, x_test,
                  split, batch_size,
                  device):

        torch.set_grad_enabled(split == 'train')
        model.train() if split == 'train' else model.eval()
        x = x_train if split == 'train' else x_test

        if batch_size <= 0 or batch_size > len(x):
            raise ValueError(
                "Batch size must be larger than 0 and smaller than sample size")

        N, D = x.size()
        B = 64  # batch size
        nsteps = math.ceil(N/B)

        loss_for_samples = torch.full((N,), torch.nan).to(device)  # N x 1
        loss_total = []

        total_samples = 0
        for step in range(nsteps):
            # fetch the next batch of data
            xb = Variable(x[step * B: step * B + B])
            self._run_batch(step, B, N, model, opt, xb, split,
                            loss_total, loss_for_samples)
            total_samples += B

        if total_samples < N:
            # fetch the remaining data
            xb = Variable(x[total_samples:])
            self._run_batch(step + 1, B, N, model, opt, xb, split,
                            loss_total, loss_for_samples)

        assert not torch.isnan(loss_for_samples).any()
        return sum(loss_total) / len(loss_total), loss_for_samples

    def _run_batch(self, step, B, N,
                   model, opt,
                   xb, split,
                   loss_total,
                   loss_for_samples):

        pred, vq_loss = model(xb)

        # if self._task == COLOR_MNIST:
        # Gaussian
        loss_each_binary = EvaluationUtils.mean_bce(pred[:, :-12], xb[:, :-12],
                                                    reduction='none')  # batch_size x D_binary
        loss_sample_binary = torch.mean(loss_each_binary, dim=1)  # batch_size x 1

        loss_each_continuos = EvaluationUtils.mean_mse(pred[:, -12:], xb[:, -12:],
                                                       reduction='none')  # batch_size x D_continuous
        loss_sample_continuous = torch.mean(loss_each_continuos, dim=1)  # batch_size x 1

        loss_sample = loss_sample_binary + loss_sample_continuous  # batch_size x 1

        reconstruction_loss = torch.mean(loss_sample)
        loss_vae = reconstruction_loss + vq_loss  # 1 x 1

        # else:
        #     # Multinomial
        #     # loss_sample = EvaluationUtils.mean_ce(pred, xb,
        #     #                                       reduction='none')  # batch_size x 1
        #     # num_drugs_taken = torch.sum(xb, dim=1)  # batch_size x 1
        #     # loss_sample = loss_sample / num_drugs_taken  # batch_size x 1
        #
        #     # Binomial
        #     pred_l = torch.sigmoid(pred)
        #     loss_sample = EvaluationUtils.mean_bce(pred_l, xb, reduction='none')
        #     loss_sample = torch.mean(loss_sample, dim=1)
        #
        #     # reconstruction_loss = EvaluationUtils.mean_ce(pred, xb,
        #     #                                               reduction='sum')
        #     reconstruction_loss = torch.mean(loss_sample)
        #     loss_vae = reconstruction_loss + vq_loss  # 1 x 1

        loss_total.append(loss_vae.item())
        # probs_sample = torch.exp(-1 * loss_sample)
        if step * B + B > N:
            loss_for_samples[step * B:] = loss_sample
        else:
            loss_for_samples[step * B: step * B + B] = loss_sample

        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss_vae.backward()
            opt.step()
