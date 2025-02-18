import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from network_architectures import CVAE as model
from .load_mnist_data import load_MNIST_data,train_val_test_split
# -----------

# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--model_path', default='saved_models/stimulus_vaes/MNIST_VAE', type=str)
p.add_argument('--mnist_data_path',default='mnist_data')
args = p.parse_args()


# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device('cpu')
print(f'USING DEVICE: {device}')


# -------------------------------------
# Load trained VAE and Model Parameters
# -------------------------------------
model_dict = torch.load(f'{args.model_path}.pt')
params = model_dict['model_params']
model = model.VAE(params).to(device)
model.load_state_dict(torch.load(model_dict['model_state_dict']))
model.eval()
dim_size = params['latent_dim_size']

# -----------------------
# Load MNIST testing data
# -----------------------
print('LOADING DATA...')
mnist_data,_ = load_MNIST_data(data_path=args.mnist_data_path,download=False)
_,_,mnist_test = train_val_test_split(mnist_data,*params['train_val_test_ratios'],seed=params['training_random_seed'])
test_loader = DataLoader(dataset=mnist_test,batch_size=len(mnist_test),shuffle=True,drop_last=True)

# --------------------------------------
# Get latent representation of test data
# --------------------------------------
for _,(data,labels) in enumerate(test_loader):
    with torch.no_grad():
        _, latent_rep, latent_logvar = model(data.to(device))

resolution = 50
Z1 = torch.linspace(latent_rep[:,0].min(),latent_rep[:,0].max(),resolution)
Z2 = torch.linspace(latent_rep[:,1].min(), latent_rep[:,1].max(),resolution)
fig,ax = plt.subplots(1,2)
ax[0].scatter(latent_rep[:,0],latent_rep[:,1],s=1)
ax[1].scatter(np.exp(0.5*latent_logvar[:,0]),np.exp(0.5*latent_logvar[:,1]),s=1)
plt.show()

# ---------------------------------
# Randomly Sample from Latent Space
# ---------------------------------
sample = torch.randn(64, dim_size).to(device)
imgs = model.decode(sample).view(-1, 1, 28, 28).detach().cpu().numpy().reshape((64,28,28))
fig,ax = plt.subplots(1,5)
for i in range(5):
    ax[i].imshow(imgs[i])
fig.suptitle('Decoded Random Points in Latent Space')
plt.show()

# ---------------------------
# Plot decoded latent tilings
# ---------------------------
fig, axs = plt.subplots(resolution, resolution, figsize=(10, 10))  # Create a 10x10 grid of subplots

with torch.no_grad():
    for i in range(resolution):
        for j in range(resolution):
            img = model.decode(torch.tensor([Z1[i], Z2[j]]).reshape(1, -1)).view(28, 28)
            axs[i, j].imshow(img.cpu().numpy(), cmap='gray')  # Plot image on the grid
            axs[i, j].axis('off')  # Turn off axis labels

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()




'''
# REMINDER THAT SCALING DOESN'T ALWAYS CARRY OVER (warning to future me)
for i, (image,label) in enumerate(train_data):
    X1 = image[0]
    #the first index is batch size (set to 1 here) so having X1=image[0] produces
    #the same values as X1=image, just without the batch index.
    X2 = train_data.data[i]
    fig,ax = plt.subplots(1,3)
    ax[0].imshow(X1)
    ax[1].imshow(X2)
    ax[2].imshow(X1-X2)
    plt.show()
    breakpoint()
'''
