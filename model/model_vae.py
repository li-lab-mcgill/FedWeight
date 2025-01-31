import torch
import torch.nn as nn
from typing import Tuple


class VAE(nn.Module):

    """ Initialize """

    def __init__(self, in_features: int,
                 hidden_sizes: str,
                 latent_dim: int) -> None:

        super(VAE, self).__init__()

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

        # Latent space layers
        self._mu_layer = nn.Linear(encoder_hiddens[-1], latent_dim)
        self._logvar_layer = nn.Linear(encoder_hiddens[-1], latent_dim)

        # Decoder layers
        self._decoder = []
        decoder_hiddens = [latent_dim] + encoder_hiddens[::-1]

        for h0, h1 in zip(decoder_hiddens, decoder_hiddens[1:]):
            self._decoder.extend([
                nn.Linear(h0, h1),
                nn.Dropout(0.3),
                nn.ReLU(),
            ])
        self._decoder.pop()  # pop the last ReLU for the output layer
        self._decoder.pop()  # pop the last Dropout for the output layer
        self._decoder.append(nn.Sigmoid())
        self._decoder = nn.Sequential(*self._decoder)

        print("Encoder: {}".format(self._encoder))
        print("Mu layer: {}".format(self._mu_layer))
        print("Log var layer: {}".format(self._logvar_layer))
        print("Decoder: {}".format(self._decoder))

    """ Public method """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        reconstructed = self._decode(z)
        return reconstructed, mu, logvar

    """ Private method """

    def _encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encoder(x)
        mu = self._mu_layer(hidden)
        logvar = self._logvar_layer(hidden)
        return mu, logvar

    def _reparameterize(self, mu: torch.Tensor,
                        logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        reconstructed = self._decoder(z)
        return reconstructed
