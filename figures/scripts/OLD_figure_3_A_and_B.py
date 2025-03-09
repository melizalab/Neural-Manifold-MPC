import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from neuron_vae_scripts.load_SNN_data_for_VAE import *
from latent_dynamics_model.time_series_loader import TimeSeriesDataset
from network_architectures.latent_linear_dynamics import LDM
from latent_dynamics_model.forecasting_functions import *

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--model_path',type=str,default='saved_models/latent_dynamics_models/LDM_prob_0.2_sample_0')
p.add_argument('--Z_out_path',default='assimilation_data/Z_assimilation',type=str)
args = p.parse_args()

# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')

# ---------
# Load LDM
# ---------
print('Loading LDM weights...')
ldm_dict = torch.load(f'{args.model_path}.pt')
ldm = LDM(ldm_dict["snn_params"]['snn_vae_model_path'],ldm_dict["mnist_params"]['mnist_cvae_model_path']).to(device)
ldm.load_state_dict(ldm_dict['model_state_dict'])
ldm.eval()

# ------------------------
# Load V Assimilation Data
# ------------------------
V = np.load(ldm_dict['mnist_params']['V_in_path'] ,allow_pickle=True)[()]
V_train = V['V_train']
V_test = V['V_test']
batch_size = len(V_test)

# -------------
# Load SNN Data
# -------------
print('Loading data...')
full_data = np.load(f"{ldm_dict['snn_params']['X_assimilation_path']}.npy",allow_pickle=True)[()]
train_spikes = concatenate_spikes(spike_data=full_data['X_train'],
                                  measurement_indxs_path=ldm_dict['snn_params']['measurement_indxs_path'],
                                  prob=ldm_dict['prob_of_measurement'],
                                  sample=ldm_dict['sample_number'])

test_spikes = concatenate_spikes(spike_data=full_data['X_test'],
                                 measurement_indxs_path=ldm_dict['snn_params']['measurement_indxs_path'],
                                 prob=ldm_dict['prob_of_measurement'],
                                 sample=ldm_dict['sample_number'])

train_data_set = TimeSeriesDataset(train_spikes,V_train)
test_data_set = TimeSeriesDataset(test_spikes,V_test)
train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

# True Latent States
Z_train_actual,_,_= ldm.encode_x(torch.from_numpy(train_spikes).to(torch.float32).to(device))
Z_test_actual,_,_= ldm.encode_x(torch.from_numpy(test_spikes).to(torch.float32).to(device))
V_train_tensor = torch.from_numpy(V_train).to(torch.float32).to(device)
V_test_tensor = torch.from_numpy(V_test).to(torch.float32).to(device)

Z_data = {'Z_train':Z_train_actual.detach().cpu().numpy(),'Z_test':Z_test_actual.detach().cpu().numpy()}
print('Saving Z assimilation data...')
np.save(args.Z_out_path+f"/prob_{ldm_dict['prob_of_measurement']}_sample_{ldm_dict['sample_number']}.npy",Z_data)






# Open Loop Forecasts (Latent States)
AB_weights = ldm.state_dict()['AB_dynamics.weight']
A = AB_weights[:,:2]
B = AB_weights[:,2:]
n_steps=batch_size-1

def Z_forecast(Z_data,V_data,n_steps):
    Z_hat = torch.zeros((n_steps,2)).to(device)
    Z_hat[0] = Z_data[0]
    for i in range(1,n_steps):
        Z_hat[i] = ldm.forward_dynamics(Z_hat[i-1],V_data[i-1])
    return Z_hat

def Z_corr(Z_actual,Z_hat):
    z1_corr = pearsonr(Z_actual[1:, 0], Z_hat[:, 0])[0]
    z2_corr = pearsonr(Z_actual[1:, 1], Z_hat[:, 1])[0]
    return z1_corr,z2_corr

print('Forecasting training data...')
Z_hat_train = Z_forecast(Z_train_actual,V_train_tensor,n_steps).detach().cpu().numpy()
Z_train_corrs = Z_corr(Z_train_actual.detach().cpu().numpy(),Z_hat_train)
print(f'Correlations ---> z1: {Z_train_corrs[0]:.2f}, z2: {Z_train_corrs[1]:.2f}')


print('Forecasting testing data...')
Z_hat_test = Z_forecast(Z_test_actual,V_test_tensor,n_steps).detach().cpu().numpy()
Z_test_corrs = Z_corr(Z_test_actual.detach().cpu().numpy(),Z_hat_test)
print(f'Correlations ---> z1: {Z_test_corrs[0]:.2f}, z2: {Z_test_corrs[1]:.2f}')


Z_train_actual = Z_train_actual.detach().cpu().numpy()
Z_test_actual = Z_test_actual.detach().cpu().numpy()

fig,ax = plt.subplots(2,2,sharex=True,sharey=True,figsize=(20,4))
# Train Z1
ax[0,0].plot(Z_train_actual[1:,0],color='black',alpha=0.7)
ax[0,0].plot(Z_hat_train[:,0],color='red',alpha=0.7)
# Train Z2
ax[1,0].plot(Z_train_actual[1:,1],color='black',alpha=0.7)
ax[1,0].plot(Z_hat_train[:,1],color='red',alpha=0.7)

# Test Z1
ax[0,1].plot(Z_test_actual[1:,0],color='black',alpha=0.7)
ax[0,1].plot(Z_hat_test[:,0],color='red',alpha=0.7)
# Test Z2
ax[1,1].plot(Z_test_actual[1:,1],color='black',alpha=0.7)
ax[1,1].plot(Z_hat_test[:,1],color='red',alpha=0.7)

# Labels
ax[0,0].set_ylabel('Z1')
ax[1,0].set_ylabel('Z2')
ax[1,0].set_xlabel('Time (ms)')
ax[1,1].set_xlabel('Time (ms)')

# Titles
ax[0,0].set_title(f'Train Z1 Corr:{Z_train_corrs[0]:.2f}')
ax[1,0].set_title(f'Train Z2 Corr:{Z_train_corrs[1]:.2f}')
ax[0,1].set_title(f'Test Z1 Corr:{Z_test_corrs[0]:.2f}')
ax[1,1].set_title(f'Test Z2 Corr:{Z_test_corrs[1]:.2f}')

ax[0,0].set_yticks([-0.1, 0 , 0.1])
ax[1,0].set_yticks([-0.1, 0 , 0.1])
ax[0,0].set_ylim([-0.15,0.15])
ax[1,0].set_ylim([-0.15,0.15])
plt.tight_layout()
plt.savefig(f'figures/raw_figures/Figure_3_Z_forecasts.pdf')


# Z Scatter Plot
fig,ax = plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))
ax[0].scatter(Z_train_actual[:,0],Z_train_actual[:,1],s=.1,alpha=.5)
ax[1].scatter(Z_test_actual[:,0],Z_test_actual[:,1],s=.1,alpha=.5)
ax[0].set_yticks([-0.1, 0 , 0.1])
ax[0].set_ylim([-0.11,0.11])
ax[0].set_xticks([-0.1, 0 , 0.1])
ax[0].set_xlim([-0.11,0.11])
ax[0].set_title('Training')
ax[1].set_title('Testing')
ax[0].set_xlabel('Z1')
ax[0].set_ylabel('Z2')
ax[1].set_xlabel('X1')
plt.savefig('figures/raw_figures/Figures_3_Z_scatter.pdf')



def X_forecast(X_data,V_data,n_steps):
    X_hat = torch.zeros((n_steps,122)).to(device)
    X_hat[0] = X_data[0]
    for i in range(1,n_steps):
        X_hat[i],_,_,_,_,_ = ldm(X_hat[i-1].reshape(1,-1),V_data[i-1].reshape(1,-1))
    return X_hat


print('Train forecast...')
X_hat_train = X_forecast(torch.from_numpy(train_spikes),V_train_tensor,n_steps=n_steps)
print('Test forecast...')
X_hat_test = X_forecast(torch.from_numpy(test_spikes),V_test_tensor,n_steps=n_steps)

np.save('figures/x_forecast_dict.npy',{'Train':{'spikes':train_spikes,'forecast':X_hat_train},'Test':{'spikes':test_spikes,'forecast':X_hat_test}})


fig,ax = plt.subplots(2,2,sharey=True,sharex=True)
ax[0,0].imshow(train_spikes[29000:30000].T,aspect='auto')
ax[1,0].imshow(X_hat_train[29000:30000].detach().cpu().T,aspect='auto')

ax[0,1].imshow(test_spikes[19000:20000].T,aspect='auto')
ax[1,1].imshow(X_hat_test[19000:20000].detach().cpu().T,aspect='auto')
ax[0,0].set_xticks([0,500,1000])

plt.show()
breakpoint()