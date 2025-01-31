import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):

    def __init__(self, num_embeddings: int,
                 embedding_dim: int,
                 beta: float,
                 device: torch.device) -> None:

        super(VectorQuantizer, self).__init__()

        self._num_embeddings = num_embeddings  # K
        self._embedding_dim = embedding_dim  # D
        self._beta = beta  # Commitment loss coefficient
        self._decay = 0.99
        self._epsilon = 1e-5
        self.register_buffer('_ema_cluster_size',
                             torch.zeros(num_embeddings))  # 1 x K
        self._ema_w = nn.Parameter(torch.Tensor(
            num_embeddings, self._embedding_dim))  # K x D
        self._ema_w.data.normal_()
        self._device = device

        # Codebook
        self._embedding = nn.Embedding(self._num_embeddings,  # K x D
                                       self._embedding_dim)
        self._embedding.weight.data.uniform_(-1 / self._num_embeddings,  # 1/K to make sure the integral of PDF is 1
                                             1 / self._num_embeddings)

    def forward(self, inputs: torch.Tensor):

        # Euclidean distances
        # print(inputs.shape)  # N x D

        distances = (torch.sum(inputs**2, dim=1, keepdim=True)  # N x K
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(inputs, self._embedding.weight.t()))

        # Latent representation z = q(z|x) e.g. [0 0 1 0 0]
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # N x 1
        latent = torch.zeros(encoding_indices.shape[0],  # N x K
                             self._num_embeddings,
                             device=self._device)
        latent.scatter_(1, encoding_indices, 1)

        # Codeword
        codeword = torch.matmul(latent, self._embedding.weight)  # N x D

        # EMA
        if self.training:
            # torch.sum(latent, 0): observed cluster size per codeword (1 x K)
            # self._ema_cluster_size: smoothed cluster size per codeword
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                (1 - self._decay) * torch.sum(latent, 0)  # 1 x K

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            # Row of latent.T: Binary vector for codeword neighbours
            # 1: the encoder output is close to codeword k (inside cluster of centroid k)
            # 0: the encoder output is not close to codeword k
            # Column of inputs: Encoder output by different samples
            # dw: Sum of encoder outputs close to codeword k
            # self._ema_w: Smoothed sum of encoder outputs close to codeword k
            dw = latent.T @ inputs  # K x D
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw)
            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        # Freeze encoder output, train codebook to make sure codeword close to encoder output
        # codebook_loss = F.mse_loss(codeword, inputs.detach()) # No need if using EMA!
        # Freeze codeword, train encoder to make sure encoder output close to codeword
        commitment_loss = F.mse_loss(codeword.detach(), inputs)
        # loss = codebook_loss + self._beta * commitment_loss # No need if using EMA!
        loss = self._beta * commitment_loss

        # Copy gradients of codeword to inputs
        codeword = inputs + (codeword - inputs).detach()

        return loss, codeword


class VQVAE(nn.Module):

    def __init__(self, in_features: int,
                 hidden_sizes: str,
                 latent_dim: int,
                 beta: float,
                 device: torch.device) -> None:

        super(VQVAE, self).__init__()

        hidden_list = list(map(int, hidden_sizes.split(',')))

        # Encoder layers
        self._encoder = []
        encoder_hiddens = [in_features] + hidden_list
        for h0, h1 in zip(encoder_hiddens, encoder_hiddens[1:]):
            self._encoder.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self._encoder.pop() # pop the last ReLU for the output layer
        self._encoder = nn.Sequential(*self._encoder)

        self._vq = VectorQuantizer(latent_dim,
                                   encoder_hiddens[-1],
                                   beta, device)

        # Decoder layers
        self._decoder = []
        decoder_hiddens = encoder_hiddens[::-1]
        for h0, h1 in zip(decoder_hiddens, decoder_hiddens[1:]):
            self._decoder.extend([
                nn.Linear(h0, h1),
                nn.ReLU(),
            ])
        self._decoder.pop()  # pop the last ReLU for the output layer
        self._decoder.append(nn.Sigmoid())
        self._decoder = nn.Sequential(*self._decoder)

    def forward(self, x: torch.Tensor):

        z = self._encoder(x)
        loss, codeword = self._vq(z)
        reconstructed = self._decoder(codeword)

        return reconstructed, loss
