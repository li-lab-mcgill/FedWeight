import torch
from torch.autograd import Variable
import math

from utils.utils_eval import EvaluationUtils
from utils.constants import *


class MadeService:

    def __init__(self, task: str) -> None:
        self._task = task

    def run_made(self,
                 model, opt,
                 x_train, x_test,
                 split, batch_size,
                 samples, resample_every,
                 device):

        # enable/disable grad for efficiency of forwarding test batches
        torch.set_grad_enabled(split == 'train')
        model.train() if split == 'train' else model.eval()
        nsamples = 1 if split == 'train' else samples
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
                            nsamples, resample_every, loss_total, loss_for_samples)
            total_samples += B

        if total_samples < N:
            # fetch the remaining data
            xb = Variable(x[total_samples:])
            self._run_batch(step + 1, B, N, model, opt, xb, split,
                            nsamples, resample_every, loss_total, loss_for_samples)

        assert not torch.isnan(loss_for_samples).any()
        return sum(loss_total) / len(loss_total), loss_for_samples

    def _run_batch(self, step, B, N,
                   model, opt,
                   xb, split,
                   nsamples, resample_every,
                   loss_total, loss_for_samples):

        # get the logits, potentially run the same batch a number of times, resampling each time
        xbhat = torch.zeros_like(xb)
        for _ in range(nsamples):
            # perform order/connectivity-agnostic training by resampling the masks
            if step % resample_every == 0 or split == 'test':  # if in test, cycle masks every time
                model.update_masks()
            # forward the model
            xbhat += model(xb)
        xbhat /= nsamples

        # evaluate the binary cross entropy loss
        # if self._task == COLOR_MNIST:
        # Gaussian
        pred = xbhat
        loss_each_binary = EvaluationUtils.mean_bce(pred[:, :-12], xb[:, :-12],
                                                    reduction='none')  # batch_size x D_binary
        loss_mean_binary = torch.mean(loss_each_binary)  # 1 x 1
        loss_sample_binary = torch.mean(loss_each_binary, dim=1)  # batch_size x 1

        loss_each_continuous = EvaluationUtils.mean_mse(pred[:, -12:], xb[:, -12:],
                                                        reduction='none') # batch_size x D_continuous
        loss_mean_continuous = torch.mean(loss_each_continuous) # 1 x 1
        loss_sample_continuous = torch.mean(loss_each_continuous, dim=1) # batch_size x 1

        loss_mean = loss_mean_binary + loss_mean_continuous
        loss_sample = loss_sample_binary + loss_sample_continuous
        
        # elif self._task == BINARIZED_MNIST:
        # else:
        #
        #     # Binary
        #     pred = torch.sigmoid(xbhat)
        #     loss_each = EvaluationUtils.mean_bce(pred, xb,
        #                                          reduction='none')  # batch_size x D
        #     loss_sample = torch.mean(loss_each, dim=1)  # batch_size x 1
        #     loss_mean = EvaluationUtils.mean_bce(pred, xb)  # 1 x 1
        
        # else:
            # Multinomial
            # pred = xbhat
            # loss_sample = EvaluationUtils.mean_ce(pred, xb,
            #                                       reduction='none')  # batch_size x 1
            # num_drugs_taken = torch.sum(xb, dim=1) # batch_size x 1
            # loss_sample = loss_sample / num_drugs_taken # batch_size x 1
            # loss_mean = torch.mean(loss_sample)  # 1 x 1

        loss_total.append(loss_mean.item())
        # probs_sample = torch.exp(-1 * loss_sample)
        if step * B + B > N:
            loss_for_samples[step * B:] = loss_sample
        else:
            loss_for_samples[step * B: step * B + B] = loss_sample

        # backward/update
        if split == 'train':
            opt.zero_grad()
            loss_mean.backward()
            opt.step()
