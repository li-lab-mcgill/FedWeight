import torch
from torch import nn
from torch.autograd import Variable
import math

from utils.utils_eval import EvaluationUtils
from utils.utils_log import LogUtils
from utils.constants import *


class VaeService:

    def __init__(self, task: str) -> None:
        self._task = task

    def run_vae(self,
                model, opt,
                x_train, x_test,
                split, batch_size,
                beta, device):

        if split == 'test':
            beta = 1.0

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
        recon_loss_total = []
        kl_total = []

        total_samples = 0
        for step in range(nsteps):
            # fetch the next batch of data
            xb = Variable(x[step * B: step * B + B])
            self._run_batch(step, B, N, model, opt, xb, split, beta,
                            loss_total, loss_for_samples, recon_loss_total, kl_total)
            total_samples += B

        if total_samples < N:
            # fetch the remaining data
            xb = Variable(x[total_samples:])
            self._run_batch(step + 1, B, N, model, opt, xb, split, beta,
                            loss_total, loss_for_samples, recon_loss_total, kl_total)

        assert not torch.isnan(loss_for_samples).any()
        return sum(loss_total) / len(loss_total), loss_for_samples, sum(recon_loss_total) / len(recon_loss_total), sum(kl_total) / len(kl_total)

    def _run_batch(self, step, B, N,
                   model, opt,
                   xb, split, beta,
                   loss_total, loss_for_samples,
                   recon_loss_total, kl_total):

        pred, mu, logvar = model(xb)

        # if self._task == COLOR_MNIST:
        # Gaussian
        loss_each_binary = EvaluationUtils.mean_bce(pred[:, :-12], xb[:, :-12],
                                                    reduction='none')  # batch_size x D_binary
        loss_sample_binary = torch.mean(loss_each_binary, dim=1)  # batch_size x 1

        loss_each_continuos = EvaluationUtils.mean_mse(pred[:, -12:], xb[:, -12:],
                                                       reduction='none')  # batch_size x D_continuous
        loss_sample_continuous = torch.mean(loss_each_continuos, dim=1)  # batch_size x 1

        loss_sample = loss_sample_binary + loss_sample_continuous  # batch_size x 1

        kl_sample = -0.5 * \
            torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # batch_size x 1

        loss_vae = torch.mean(loss_sample + beta * kl_sample)  # 1 x 1
        recon_loss = torch.mean(loss_sample)
        kl_loss = torch.mean(kl_sample)

        loss_sample += kl_sample  # batch_size x 1

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
        #     # batch_size x 1
        #     kl_sample = -0.5 * \
        #         torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        #
        #     loss_vae = torch.mean(loss_sample + beta * kl_sample)  # 1 x 1
        #     recon_loss = torch.mean(loss_sample)
        #     kl_loss = torch.mean(kl_sample)
        #
        #     loss_sample += kl_sample  # batch_size x 1

        loss_total.append(loss_vae.item())
        recon_loss_total.append(recon_loss.item())
        kl_total.append(kl_loss.item())
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
