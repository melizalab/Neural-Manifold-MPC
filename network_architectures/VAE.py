import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.input_layer_size = params['input_size']
        self.latent_dim_size = params['latent_dim_size']
        self.hidden_layer_sizes = params['hidden_layer_sizes']  # List of hidden layer sizes
        self.eps_scaling = params['eps_scaling']

        # Define encoding layers
        self.encoder_layers = nn.ModuleList()
        self.encoder_bns = nn.ModuleList()  # BatchNorm layers for encoder
        prev_size = self.input_layer_size
        for h_size in self.hidden_layer_sizes:
            self.encoder_layers.append(nn.Linear(prev_size, h_size))
            self.encoder_bns.append(nn.BatchNorm1d(h_size))
            prev_size = h_size

        # Latent distribution variables
        self.enc_mean = nn.Linear(self.hidden_layer_sizes[-1], self.latent_dim_size)
        self.enc_logvar = nn.Linear(self.hidden_layer_sizes[-1], self.latent_dim_size)

        # Define decoding layers
        self.decoder_layers = nn.ModuleList()
        self.decoder_bns = nn.ModuleList()  # BatchNorm layers for decoder
        prev_size = self.latent_dim_size
        for h_size in reversed(self.hidden_layer_sizes):
            self.decoder_layers.append(nn.Linear(prev_size, h_size))
            self.decoder_bns.append(nn.BatchNorm1d(h_size))
            prev_size = h_size

        # Output layer
        self.decode_output = nn.Linear(self.hidden_layer_sizes[0], self.input_layer_size)

    # Encoding block
    def encoder(self, x):
        for layer, bn in zip(self.encoder_layers, self.encoder_bns):
            x = F.relu(bn(layer(x)))
        mu = self.enc_mean(x)
        logvar = self.enc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std) * self.eps_scaling
        z = mu + std * eps
        return z

    def decoder(self, z):
        for layer, bn in zip(self.decoder_layers, self.decoder_bns):
            z = F.relu(bn(layer(z)))  # Linear -> BatchNorm -> ReLU
        x = torch.sigmoid(self.decode_output(z))  # Use if decoded is between 0 and 1 (not normalized)
        #x = self.decode_output(z)
        return x

    # Forward pass
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        output = self.decoder(z)
        return output, mu, logvar

    # Encoder-only pass
    def encode(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    # Decoder-only pass
    def decode(self, z):
        return self.decoder(z)
