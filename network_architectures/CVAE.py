import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, params):
        super(VAE, self).__init__()
        self.latent_dim_size = params['latent_dim_size']

        # Encoder
        self.encoder_filters = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.encoder_h1 = nn.Linear(3136,256)

        # Latent space
        self.fc_mu = nn.Linear(256, self.latent_dim_size)
        self.fc_logvar = nn.Linear(256, self.latent_dim_size)

        # Decoder
        self.decoder_h1 = nn.Linear(self.latent_dim_size,256)
        self.decoder_h2 = nn.Linear(256,3136)
        self.decoder_filters = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32,1, kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder_filters(x)
        x = F.relu(self.encoder_h1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu,logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar/2)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x = F.relu(self.decoder_h1(z))
        x = F.relu(self.decoder_h2(x))
        x = self.decoder_filters(x)
        return x[:,:,:28,:28]

    def forward(self, x):
        z,mu, logvar = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


# Decoder Only
class DecoderOnly(nn.Module):
    def __init__(self, mnist_cvae):
        super(DecoderOnly, self).__init__()
        # Initialize the decoder layers from the original VAE
        self.decoder_h1 = mnist_cvae.decoder_h1
        self.decoder_h2 = mnist_cvae.decoder_h2
        self.decoder_filters = mnist_cvae.decoder_filters

    def decode(self, v):
        # Pass through the decoder layers
        v = F.relu(self.decoder_h1(v))
        v = F.relu(self.decoder_h2(v))
        u = self.decoder_filters(v)
        return u[:, :, :28, :28]