import sys
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from scipy.stats import pearsonr
from tqdm import tqdm
import copy

# ------------------------------
# Append path for custom modules
# ------------------------------
from network_architectures import VAE as model
from .load_SNN_data_for_VAE import VAE_dataloader,concatenate_spikes

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument("--X_assimilation_path", default='assimilation_data/filtered_spikes_assimilation',type=str)
p.add_argument("--measurement_indxs_path", default='assimilation_data/spikes_measurement_indxs',type=str)
p.add_argument('--prob_of_measurement',default=.2,type=float)
p.add_argument('--sample_number',default=0,type=int)
p.add_argument('--hidden_layer_sizes',nargs='+',default=[500,200,100,50,10], type = int)
p.add_argument("--latent_dim_size", default=2, type=int)
p.add_argument("--batch_size", default=256, type=int)
p.add_argument("--num_epochs", default=20, type=int)
p.add_argument('--kl_scaling',default=0.001,type=float)
p.add_argument('--learning_rate',default=5e-4,type=float)
p.add_argument('--decay_rate',default=0.99,type=float)
p.add_argument('--eps_scaling',default=1.0,type=float)
p.add_argument('--std_eps',default=1e-5,type=float)
p.add_argument('--num_examples',default=4000,type=int)
# Path to Save VAE
p.add_argument("--save_path", default='saved_models/snn_vaes/pretrained_SNN_VAE')
args = p.parse_args()

# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')

# -------------
# Load SNN Data
# -------------
print('Loading data...')
full_data = np.load(f'{args.X_assimilation_path}.npy',allow_pickle=True)[()]
measured_spikes = concatenate_spikes(spike_data=full_data['X_train'],
                                     measurement_indxs_path=args.measurement_indxs_path,
                                     prob=args.prob_of_measurement,
                                     sample=args.sample_number)

train_loader,input_size = VAE_dataloader(measured_spikes=measured_spikes,
                                         batch_size=args.batch_size,
                                         shuffle=True)
print(f'Number of neurons recorded from: {input_size}')

def normalize_batch(X_batch,std_eps=args.std_eps):
    X_mean = X_batch.mean()
    X_std = X_batch.std()
    return (X_batch-X_mean)/(X_std+std_eps)

# -------------------
# Get VAE parameters
# ------------------
params = {'input_size':input_size,
          'latent_dim_size':args.latent_dim_size,
          'hidden_layer_sizes':args.hidden_layer_sizes,
          'eps_scaling':args.eps_scaling,
          'batch_size':args.batch_size,
          'kl_scaling':args.kl_scaling,
          'num_epochs':args.num_epochs,
          'learning_rate':args.learning_rate,
          'deacy_rate':args.decay_rate,
          'std_eps':args.std_eps}

# ------------
# Instance VAE
# ------------
model = model.VAE(params).to(device)
model.train()

# -----------------
# VAE loss functions
# -----------------
def loss_function(output,x,mu,logvar,kl_scaling):
    MSE = nn.MSELoss(reduction='sum')
    recon_loss = MSE(x,output)
    kl_loss = -kl_scaling*torch.sum(1+logvar-mu**2-logvar.exp())
    return recon_loss,kl_loss

# ----------------
# Set up optimizer
# ----------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
lmbda = lambda epoch: args.decay_rate**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

# ---------
# Train VAE
# ---------
print('Training VAE...')
train_loss = []
for epoch in range(args.num_epochs):
    counter = 0
    for X in tqdm(train_loader):
        X = X.to(device)
        # Get batch and predicted values
        X_hat,mu,logvar = model(X)

        # Calculate batch loss
        recon_loss, kl_loss = loss_function(X_hat, X, mu, logvar,kl_scaling=args.kl_scaling)
        batch_loss = recon_loss+kl_loss

        # Backpropagate
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Append losses
        train_loss.append(batch_loss.detach().cpu().numpy())

    # Print accuracy
    print(f'Epoch: {epoch+1}/{args.num_epochs}, '
            f'train loss: {batch_loss:.2f}, '
            f'recon_loss: {recon_loss:.2f}, '
            f'kl_loss: {kl_loss:.7f}',
            f'lr: {optimizer.param_groups[0]["lr"]:.7f}')
    scheduler.step()
# Store last learning rate for fine-tuning in LDM
params['final_learning_rate'] = optimizer.param_groups[0]['lr']

# --------------------
# Evaluate predictions
# --------------------
model.eval()
X = normalize_batch(torch.from_numpy(measured_spikes).float().to(device))
X_hat,mu,logvar = model(X.view(X.shape[0], -1).to(device))
fig,ax = plt.subplots(2,1,sharex=True,sharey=True)
ax[0].imshow(X.detach().cpu().numpy().T,aspect='auto')
ax[1].imshow(X_hat.detach().cpu().numpy().T,aspect='auto')
plt.show()

# ----------
# Save Model
# ----------
model_save_path = f'{args.save_path}_prob_{args.prob_of_measurement}_sample_{args.sample_number}'
print(f'Saving model at: {model_save_path}')
torch.save({'model_params':params,
            'last_train_loss':train_loss[epoch],
            'model_state_dict': copy.deepcopy(model.state_dict())},
            f'{model_save_path}.pt')

