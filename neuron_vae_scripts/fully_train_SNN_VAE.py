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
p.add_argument('--hidden_layer_sizes',nargs='+',default=[1000,1000], type = int)
p.add_argument("--latent_dim_size", default=2, type=int)
p.add_argument("--batch_size", default=256, type=int)
p.add_argument("--max_num_epochs", default=200, type=int)
p.add_argument('--kl_scaling',default=0.001,type=float)
p.add_argument('--learning_rate',default=1e-3,type=float)
p.add_argument('--decay_rate',default=0.99,type=float)
p.add_argument('--eps_scaling',default=1.0,type=float)
p.add_argument('--early_stopping_threshold',default=5, type = int)
# Path to Save VAE
p.add_argument("--save_path", default='saved_models/snn_vaes/SNN_VAE')
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


# Training data
measured_spikes_train = concatenate_spikes(spike_data=full_data['X_train'],
                                     measurement_indxs_path=args.measurement_indxs_path,
                                     prob=args.prob_of_measurement,
                                     sample=args.sample_number)

train_loader,n_neurons = VAE_dataloader(measured_spikes=measured_spikes_train,
                                         batch_size=args.batch_size,
                                         shuffle=True)

# Validation data
measured_spikes_val = concatenate_spikes(spike_data=full_data['X_val'],
                                     measurement_indxs_path=args.measurement_indxs_path,
                                     prob=args.prob_of_measurement,
                                     sample=args.sample_number)

val_loader,_ = VAE_dataloader(measured_spikes=measured_spikes_val ,
                                         batch_size=args.batch_size,
                                         shuffle=True)
print(f'Number of neurons recorded from: {n_neurons}')

# -------------------
# Get VAE parameters
# ------------------
params = {'input_size':n_neurons,
          'latent_dim_size':args.latent_dim_size,
          'hidden_layer_sizes':args.hidden_layer_sizes,
          'eps_scaling':args.eps_scaling,
          'batch_size':args.batch_size,
          'kl_scaling':args.kl_scaling,
          'max_num_epochs':args.max_num_epochs,
          'learning_rate':args.learning_rate,
          'deacy_rate':args.decay_rate}

# ------------
# Instance VAE
# ------------
model = model.VAE(params).to(device)

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
val_loss = []
best_val_loss = float('inf')
for epoch in range(args.max_num_epochs):
    counter = 0
    train_epoch_loss = 0
    num_train_obs = 0
    val_epoch_loss = 0
    num_val_obs = 0
    print('Training loop...')
    model.train()
    for X in tqdm(train_loader):
        optimizer.zero_grad()
        # Get batch and predicted values
        X_hat,mu,logvar = model(X.to(device))

        # Calculate batch loss
        recon_loss, kl_loss = loss_function(X_hat, X.reshape(args.batch_size, -1).to(device), mu, logvar,kl_scaling=args.kl_scaling)
        batch_loss = recon_loss+kl_loss

        # Backpropagate
        batch_loss.backward()
        optimizer.step()

        # Append losses
        train_loss.append(batch_loss.detach().cpu().numpy())
        train_epoch_loss+=train_loss[-1]
        num_train_obs+=len(X)

    train_mean_loss = train_epoch_loss/num_train_obs
    print('Validation loop...')
    model.eval()
    with torch.inference_mode():
        for X_val in tqdm(val_loader):
            # Get batch and predicted values
            X_hat_val, mu_val, logvar_val = model(X_val.to(device))

            # Calculate batch loss
            recon_loss_val, kl_loss_val = loss_function(X_hat_val, X_val.reshape(args.batch_size, -1).to(device), mu_val, logvar_val,kl_scaling=args.kl_scaling)
            batch_loss_val = recon_loss_val+kl_loss_val

            # Append losses
            val_loss.append(batch_loss_val.detach().cpu().numpy())
            val_epoch_loss+=val_loss[-1]
            num_val_obs+=len(X_val)
        val_mean_loss = val_epoch_loss/num_val_obs
    if val_mean_loss < best_val_loss:
         best_val_loss = val_mean_loss
         print(f"New best model found at epoch {epoch + 1}!")
         early_stop_counter = 0
         params['stopped_at_epoch']=epoch+1
         params['last_train_loss'] = train_mean_loss
         params['last_val_loss'] = val_mean_loss
         params['model_state_dict'] = copy.deepcopy(model.state_dict())
         model_save_path = f'{args.save_path}_prob_{args.prob_of_measurement}_sample_{args.sample_number}'
         torch.save(params,f'{model_save_path}.pt')
    else:
        early_stop_counter += 1
        print(f"No improvement of validation loss in {early_stop_counter} epoch(s).")
    # Early stopping check
    if early_stop_counter >= args.early_stopping_threshold:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break
    # Print accuracy
    print(f'Epoch:{epoch+1}/{args.max_num_epochs}')
    print(f'Training Losses:  ',
            f'mean epoch loss: {train_mean_loss:.2f}',
            f'batch loss: {batch_loss.item():.2f}',
            f'recon loss: {recon_loss.item():.2f}',
            f'kl loss: {kl_loss.item():.2f}',
            f'lr: {optimizer.param_groups[0]["lr"]:.7f}')
    print(f'Validation Losses:',
            f'mean epoch loss: {val_mean_loss:.2f}',
            f'batch loss: {batch_loss_val.item():.2f}',
            f'recon loss: {recon_loss_val.item():.2f}',
            f'kl loss: {kl_loss_val.item():.2f}')
    scheduler.step()
breakpoint()