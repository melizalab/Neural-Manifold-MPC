import torch
import torch.nn as nn
import numpy as np
import sys
from network_architectures import VAE as vae
from network_architectures import CVAE as CVAE

class LDM(nn.Module):
    def __init__(self, snn_vae_path, mnist_cvae_path):
        super().__init__()
        # Load SNN VAE
        snn_vae_dict = torch.load(f'{snn_vae_path}.pt')
        snn_vae = vae.VAE(params=snn_vae_dict['model_params'])
        snn_vae.load_state_dict(snn_vae_dict['model_state_dict'])
        self.snn_vae = snn_vae
        # Load MNIST CVAE decoder
        mnist_cvae_dict = torch.load(f'{mnist_cvae_path}.pt')
        mnist_cvae = CVAE.VAE(params=mnist_cvae_dict['model_params'])
        mnist_cvae.load_state_dict(mnist_cvae_dict['model_state_dict'])
        self.u_decoder = CVAE.DecoderOnly(mnist_cvae)
        # Freeze the u_decoder weights
        for param in self.u_decoder.parameters():
            param.requires_grad = False
        # Latent Dynamics Matrix
        self.AB_dynamics = nn.Linear(4, 2,bias=False)

    def encode_x(self,x_n):
        z_n, mu_n, logvar_n = self.snn_vae.encode(x_n)
        return z_n,mu_n,logvar_n
    def forward_dynamics(self,z_n,v_n):
        z_np1 = self.AB_dynamics(torch.cat((z_n,v_n), dim=-1))
        return z_np1
    def decode_z(self,z_np1):
        x_np1 = self.snn_vae.decode(z_np1)
        return x_np1
    # Forward pass
    def forward(self,x_n,v_n):
        z_n,mu_n,logvar_n = self.encode_x(x_n)
        z_np1 = self.forward_dynamics(z_n,v_n)
        x_np1 = self.decode_z(z_np1)
        u_n = self.u_decoder.decode(v_n)
        return x_np1,u_n,z_np1,z_n,mu_n,logvar_n
