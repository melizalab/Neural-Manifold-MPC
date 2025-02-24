import sys
import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
import copy

from neuron_vae_scripts.load_SNN_data_for_VAE import *
from network_architectures.latent_linear_dynamics import LDM
from .time_series_loader import TimeSeriesDataset_w_Prediction

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--save_path',type=str,default='saved_models/latent_dynamics_models/LDM')
# SNN VAE args
p.add_argument('--snn_vae_model_path',type=str,default='saved_models/snn_vaes/pretrained_SNN_VAE')
p.add_argument('--X_assimilation_path',type=str,default='assimilation_data/filtered_spikes_assimilation')
p.add_argument("--measurement_indxs_path", default='assimilation_data/spikes_measurement_indxs',type=str)
p.add_argument('--prob_of_measurement',default=.2,type=float)
p.add_argument('--sample_number',default=0,type=int)
# MNIST VAE args
p.add_argument('--mnist_cvae_model_path',type=str,default='saved_models/stimulus_vaes/MNIST_VAE')
p.add_argument('--V_in_path',type=str,default='assimilation_data/V_assimilation.npy')
# Autoregressive predictions args
p.add_argument('--n_step_prediction',default=10,type=int)
# Learning hyperparameters
p.add_argument('--decay_rate',type=float,default=.99)
p.add_argument('--batch_size',type=int,default=256)
p.add_argument('--max_num_epochs',type=int,default=200)
p.add_argument('--early_stopping_threshold',type=int,default=5)
p.add_argument('--kl_scaling',type=float,default=0.01)
p.add_argument('--learning_rate',type=float,default=3e-4)
args = p.parse_args()

# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')

# ------------------------------------
# Load LDM and get training parameters
# ------------------------------------
print('BUILDING NETWORK...')
snn_vae_model_path_full = args.snn_vae_model_path+f'_prob_{args.prob_of_measurement}_sample_{args.sample_number}'
ldm = LDM(snn_vae_model_path_full,args.mnist_cvae_model_path).to(device)
params = {'snn_params':
            {'snn_vae_model_path':snn_vae_model_path_full,
             'X_assimilation_path':args.X_assimilation_path,
             'measurement_indxs_path':args.measurement_indxs_path},
        'prob_of_measurement':args.prob_of_measurement,
        'sample_number':args.sample_number,
        'mnist_params':
            {'mnist_cvae_model_path':args.mnist_cvae_model_path,
             'V_in_path':args.V_in_path},
        'autoregressive_params':
            {'n_step_prediction':args.n_step_prediction},
        'training_params':
            {'learning_rate':args.learning_rate,
             'decay_rate':args.decay_rate,
             'max_num_epochs':args.max_num_epochs,
             'batch_size':args.batch_size,
             'kl_scaling':args.kl_scaling}}
snn_vae_dict = torch.load(f'{snn_vae_model_path_full}.pt')
# ------------------------
# Load V Assimilation Data
# ------------------------
V = np.load(args.V_in_path,allow_pickle=True)[()]
V_train = V['V_train']
V_val = V['V_val']

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
train_data_set = TimeSeriesDataset_w_Prediction(measured_spikes_train,V_train,args.n_step_prediction)
train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True)
n_neurons = measured_spikes_train.shape[1]

# Validation data
measured_spikes_val = concatenate_spikes(spike_data=full_data['X_val'],
                                     measurement_indxs_path=args.measurement_indxs_path,
                                     prob=args.prob_of_measurement,
                                     sample=args.sample_number)
val_data_set = TimeSeriesDataset_w_Prediction(measured_spikes_val,V_val,args.n_step_prediction)
val_loader = DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False)

# ----------------------------
# Loss Functions and Optimizer
# ----------------------------
print('INITIALIZING LOSS AND OPTIMIZER...')
def BCE_loss(x_hat,x,mu,logvar,kl_scaling):
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
    kl_loss = -kl_scaling * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / x.size(0)
    return recon_loss,kl_loss

def MSE_loss(predictions,labels):
    MSE = nn.MSELoss(reduction='sum')
    pred_loss = MSE(predictions,labels)
    return pred_loss

optimizer = torch.optim.Adam(ldm.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
lmbda = lambda epoch: args.decay_rate**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)


def prediction_loop(x_n,v_n,x_future,v_future,model):
        # Predict x_{n+1}
        x_np1_hat,u_n_hat,z_np1_hat,z_n_hat,mu_n,logvar_n = model(x_n, v_n)

        # Predict z_{n+1}:z_{n+k}
        Z_HATs = [z_n_hat]  # Initialize with the first prediction
        for n in range(1, args.n_step_prediction + 1):
            z_next = model.forward_dynamics(Z_HATs[-1], v_future[:, n - 1])
            Z_HATs.append(z_next)
        Z_HATs = torch.stack(Z_HATs, dim=1)

        # Get Actual z_futures
        '''
        TODO: Maybe change this to the expectation instead?
        '''
        Z_ACTUALs,_,_ = model.encode_x(x_future.view(-1,n_neurons))
        Z_ACTUALs = Z_ACTUALs.view(Z_HATs.shape[0],args.n_step_prediction+1,-1)

        # BCE losses
        recon_loss, kl_loss = BCE_loss(x_hat=x_np1_hat,
                                       x=x_future[:,0,:],
                                       mu=mu_n,
                                       logvar=logvar_n,
                                       kl_scaling=args.kl_scaling)
        # Forecast prediction loss
        pred_loss = MSE_loss(Z_HATs[:,1:,:],Z_ACTUALs[:,1:,:])
        '''
        AB_weights = model.state_dict()['AB_dynamics.weight']
        A = AB_weights[:,:2]
        B = AB_weights[:,2:]
        '''
        return recon_loss,kl_loss,pred_loss


# ---------
# Train LDM
# ---------
# Track net performance
recon_losses = []
kl_losses = []
pred_losses = []
best_val_loss = float('inf')
# Training loop
ldm.train()
print('TRAINING NETWORK...')
for epoch in range(args.max_num_epochs):
    train_epoch_loss = 0
    num_train_obs = 0
    val_epoch_loss = 0
    num_val_obs = 0
    # Training Loop
    print('training loop...')
    ldm.train()
    for (x_n,v_n,x_future,v_future) in tqdm(train_loader):
        optimizer.zero_grad()
        # Batch sample x_n, v_n, x_future
        x_n = x_n.to(device)
        v_n = v_n.to(device)
        x_future = x_future.to(device)
        v_future = v_future.to(device)
        # training predicitons
        recon_loss,kl_loss,pred_loss = prediction_loop(x_n,v_n,x_future,v_future,ldm)
        train_batch_loss = recon_loss+kl_loss+pred_loss
        # Log losses
        recon_losses.append(recon_loss.item())
        kl_losses.append(kl_loss.item())
        pred_losses.append(pred_loss.item())
        # Gradient calculation + weight update
        train_batch_loss.backward()
        optimizer.step()
        train_epoch_loss+= train_batch_loss.item()
        num_train_obs+= len(x_n)
    train_mean_loss = train_epoch_loss/num_train_obs
    # Validation Loop
    print('validation loop...')
    ldm.eval()
    with torch.inference_mode():
        for (x_n,v_n,x_future,v_future) in tqdm(val_loader):
            # Batch sample x_n, v_n, x_future
            x_n = x_n.to(device)
            v_n = v_n.to(device)
            x_future = x_future.to(device)
            v_future = v_future.to(device)
            # training predicitons
            val_recon_loss,val_kl_loss,val_pred_loss= prediction_loop(x_n,v_n,x_future,v_future,ldm)
            val_batch_loss = val_recon_loss+val_kl_loss+val_pred_loss
            val_epoch_loss+= val_batch_loss.item()
            num_val_obs+=len(x_n)
        val_mean_loss = val_epoch_loss/num_val_obs
    
    if val_mean_loss < best_val_loss:
         best_val_loss = val_mean_loss
         print(f"New best model found at epoch {epoch + 1}!")
         early_stop_counter = 0
         params['training_params']['stopped_at_epoch']=epoch+1
         params['training_params']['last_train_loss'] = train_mean_loss
         params['training_params']['last_val_loss'] = val_mean_loss
         params['model_state_dict'] = copy.deepcopy(ldm.state_dict())
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
            f'batch loss: {train_batch_loss.item():.2f}',
            f'recon loss: {recon_loss.item():.2f}',
            f'kl loss: {kl_loss.item():.2f}',
            f'pred loss: {pred_loss.item():.2f}',
            f'lr: {optimizer.param_groups[0]["lr"]:.7f}')
    print(f'Validation Losses:',
            f'mean epoch loss: {val_mean_loss:.2f}',
            f'batch loss: {val_batch_loss.item():.2f}',
            f'recon loss: {val_recon_loss.item():.2f}',
            f'kl loss: {val_kl_loss.item():.2f}',
            f'pred loss: {val_pred_loss.item():.2f}')
    scheduler.step()