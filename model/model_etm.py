import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ETM(nn.Module):
    def __init__(self,
                 input_size,
                 dedicated_hospital_ids=None,
                 run_with_fl=True):
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = 18
        self.vocab_size = input_size
        self.rho_size = 256

        self.rho = nn.Parameter(torch.randn(input_size, self.rho_size))  # D x L

        self.run_with_fl = run_with_fl
        if not run_with_fl:
            self.dedicated_hospital_ids = list(
                map(float, dedicated_hospital_ids.split(',')))
            self.batch_bias = nn.Parameter(
                torch.randn(len(self.dedicated_hospital_ids), input_size))  # H x D

        ## define the matrix containing the topic embeddings
        self.alphas = nn.Linear(self.rho_size,
                                self.num_topics,
                                bias=False)  # nn.Parameter(torch.randn(rho_size, num_topics))

        ## define variational distribution for \theta_{1:D} via amortizartion
        # self.mu_q_theta = nn.Linear(16, self.num_topics, bias=True)
        # self.logsigma_q_theta = nn.Linear(16, self.num_topics, bias=True)

        q_theta_input = input_size if run_with_fl else input_size + len(self.dedicated_hospital_ids)
        # self.q_theta = nn.Sequential(
        #     nn.Linear(q_theta_input, 64),
        #     nn.ReLU(True),
        #     nn.Linear(64, 32),
        #     nn.ReLU(True),
        #     nn.Linear(32, 16),
        #     nn.ReLU(True),
        # )
        self.q_theta = nn.Sequential(
            nn.Linear(q_theta_input, 256),
            nn.ReLU(True),
            nn.Linear(256, 16),
            nn.ReLU(True),
        )
        self.mu_q_theta = nn.Linear(16, self.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(16, self.num_topics, bias=True)

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        # if self.training:
        #
        # else:
        #     return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul_(std).add_(mu)

    def encode(self, bows):
        """Returns paramters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)

        # KL[q(theta)||p(theta)] = lnq(theta) - lnp(theta)
        kl_theta = -0.5 * torch.mean(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1)  # N x 1
        return mu_theta, logsigma_theta, kl_theta

    def get_beta(self):
        ## softmax over vocab dimension
        beta = F.softmax(self.alphas(self.rho), dim=0).transpose(1, 0)
        return beta

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)

        z = self.reparameterize(mu_theta, logsigma_theta)

        theta = F.softmax(z, dim=-1)
        return theta, kld_theta

    def decode(self, theta, beta, batch_bias=None):
        recon_logit = theta @ beta

        if batch_bias is not None:
            recon_logit += batch_bias
            preds = F.log_softmax(recon_logit, dim=-1)
        else:
            preds = torch.log(recon_logit + 1e-6)

        return preds

    def forward(self, X):
        ## get \theta

        if not self.run_with_fl:
            hospital_ids = X[:, :len(self.dedicated_hospital_ids)]  # N x Hospital_ids
            X_original = X[:, len(self.dedicated_hospital_ids):]  # N x D

            sums = X_original.sum(1).unsqueeze(1)
            X_normalized = X_original / sums  # N x D

            X_concat = torch.cat((hospital_ids, X_normalized), 1)  # N x (D + Hospital_ids)

            theta, kld_theta = self.get_theta(X_concat)  # N x K
            ## get \beta
            beta = self.get_beta()  # K x D

            hospital_indices = torch.argmax(hospital_ids, dim=1)  # N x 1

            batch_bias = self.batch_bias[hospital_indices]  # N x D where self.batch_bias has shape H x D

            ## get prediction loss
            preds = self.decode(theta, beta, batch_bias)  # N x D

            recon_loss = -torch.mean(X_original * preds, dim=-1)  # N x 1

        else:
            sums = X.sum(1).unsqueeze(1)
            X_normalized = X / sums  # N x D
            theta, kld_theta = self.get_theta(X_normalized)  # N x K

            ## get \beta
            beta = self.get_beta()  # K x D

            ## get prediction loss
            preds = self.decode(theta, beta)  # N x D
            recon_loss = -torch.mean(X * preds, dim=-1)  # N x 1

        return recon_loss, kld_theta, beta
