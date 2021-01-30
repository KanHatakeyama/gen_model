"""
Simple autoencoder and variational autoencoder classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'

# variable autoencoder


class VAE(nn.Module):
    def __init__(self, z_dim, input_dim=28*28):
        super(VAE, self).__init__()
        self.dense_enc1 = nn.Linear(input_dim, 200)
        self.dense_enc2 = nn.Linear(200, 200)
        self.dense_encmean = nn.Linear(200, z_dim)
        self.dense_encvar = nn.Linear(200, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 200)
        self.dense_dec2 = nn.Linear(200, 200)
        self.dense_dec3 = nn.Linear(200, input_dim)

    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        var = F.softplus(self.dense_encvar(x))
        return mean, var

    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape).to(device)
        return mean + torch.sqrt(var) * epsilon

    def _decoder(self, z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = torch.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, z

    def loss(self, x):
        mean, var = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean**2 - var))
        z = self._sample_z(mean, var)
        y = self._decoder(z)
        reconstruction = torch.mean(
            torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
        lower_bound = [-KL, reconstruction]
        return -sum(lower_bound)


# autoencoder
class AE(nn.Module):
    def __init__(self, z_dim, input_dim=28*28, batch_norm=True):
        super(AE, self).__init__()
        self.dense_enc1 = nn.Linear(input_dim, 200)
        self.dense_enc2 = nn.Linear(200, 200)
        self.dense_encmean = nn.Linear(200, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 200)
        self.dense_dec2 = nn.Linear(200, 200)
        self.dense_dec3 = nn.Linear(200, input_dim)
        self.batch_norm = batch_norm
        
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(input_dim)

    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        return mean

    def _decoder(self, z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = torch.sigmoid(self.dense_dec3(x))

        if self.batch_norm:
            x = self.bn1(x)
        return x

    def forward(self, x):
        mean = self._encoder(x)
        x = self._decoder(mean)
        return x, None

    def loss(self, x):
        loss = nn.MSELoss()
        mean = self._encoder(x)
        y = self._decoder(mean)

        return loss(x, y)
