import torch.nn as nn


class ModelAutoEncoder(nn.Module):
    def __init__(self, input_size):
        super(ModelAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),
        )

        # Decoder part
        self.decoder = nn.Sequential(
            # nn.Linear(16, 32),
            # nn.ReLU(True),
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstruct = self.decoder(latent)
        return reconstruct, latent
