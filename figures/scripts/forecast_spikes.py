import torch
import argparse
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm

from neuron_vae_scripts.load_SNN_data_for_VAE import *
from latent_dynamics_model.time_series_loader import TimeSeriesDataset
from network_architectures.latent_linear_dynamics import LDM
from latent_dynamics_model.forecasting_functions import *

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--model_path',type=str,default='saved_models/latent_dynamics_models')
p.add_argument('--sample_number',type=int,default=0)
p.add_argument('--prob_of_measurement',type=float,default=0.2)
args = p.parse_args()



# ---------
# Load LDM
# ---------
print('Loading LDM weights...')
ldm_dict = torch.load(f'{args.model_path}/LDM_prob_{args.prob_of_measurement}_sample_{args.sample_number}.pt')
ldm = LDM(ldm_dict["snn_params"]['snn_vae_model_path'],ldm_dict["mnist_params"]['mnist_cvae_model_path'])
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
full_data = np.load(f"{ldm_dict['snn_params']['X_assimilation_path']}.npy",allow_pickle=True)[()]
train_spikes = concatenate_spikes(spike_data=full_data['X_train'],
                                  measurement_indxs_path=ldm_dict['snn_params']['measurement_indxs_path'],
                                  prob=ldm_dict['prob_of_measurement'],
                                  sample=ldm_dict['sample_number'])

test_spikes = concatenate_spikes(spike_data=full_data['X_test'],
                                 measurement_indxs_path=ldm_dict['snn_params']['measurement_indxs_path'],
                                 prob=ldm_dict['prob_of_measurement'],
                                 sample=ldm_dict['sample_number'])
n_neurons = train_spikes.shape[1]
train_data_set = TimeSeriesDataset(train_spikes,V_train)
test_data_set = TimeSeriesDataset(test_spikes,V_test)
train_loader = DataLoader(train_data_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=False)

# True Latent States
Z_train,_,_= ldm.encode_x(torch.from_numpy(train_spikes).to(torch.float32))
Z_test,_,_= ldm.encode_x(torch.from_numpy(test_spikes).to(torch.float32))
V_train_tensor = torch.from_numpy(V_train).to(torch.float32)
V_test_tensor = torch.from_numpy(V_test).to(torch.float32)

# Open Loop Forecasts (Latent States)
AB_weights = ldm.state_dict()['AB_dynamics.weight']
A = AB_weights[:,:2]
B = AB_weights[:,2:]
n_steps=batch_size-1

def forecast(Z_data,X_data,V_data,n_steps):
    Z_hat = torch.zeros((n_steps,2))
    X_hat = torch.zeros((n_steps,n_neurons))
    Z_hat[0] = Z_data[0]
    X_hat[0] = torch.from_numpy(X_data[0]).to(torch.float32)
    for i in tqdm(range(1,n_steps)):
        Z_hat[i] = ldm.forward_dynamics(Z_hat[i-1],V_data[i-1])
        X_hat[i],_,_,_,_,_ = ldm(X_hat[i-1].reshape(1,-1),V_data[i-1].reshape(1,-1))
    return Z_hat,X_hat

Z_hat_train,X_hat_train = forecast(Z_data=Z_train,X_data=train_spikes,V_data=V_train_tensor,n_steps=Z_train.shape[0])
Z_hat_test,X_hat_test = forecast(Z_data=Z_test,X_data=test_spikes,V_data=V_test_tensor,n_steps=Z_test.shape[0])

# Get X for decoding z correlations
X_train_decoded = ldm.decode_z(Z_hat_train)
X_test_decoded = ldm.decode_z(Z_hat_test)


# Get correlations
Z_hat_train = Z_hat_train.detach().cpu().numpy()
Z_hat_test = Z_hat_test.detach().cpu().numpy()
Z_train = Z_train.detach().cpu().numpy()
Z_test = Z_test.detach().cpu().numpy()

data_dict = {'X_train':train_spikes,
             'X_test':test_spikes,
             'Z_train':Z_train,
             'Z_test':Z_test,
             'Z_hat_train':Z_hat_train,
             'Z_hat_test':Z_hat_test,
             'X_hat_train':X_hat_train.detach().cpu().numpy(),
             'X_hat_test':X_hat_test.detach().cpu().numpy(),
             'X_train_decoded':X_train_decoded.detach().cpu().numpy(),
             'X_test_decoded':X_test_decoded.detach().cpu().numpy()}
np.save('figures/data/spikes_and_latents_for_prob_0.2_sample_0.npy',data_dict)
breakpoint()
z1_train_corr = pearsonr(Z_train[:,0].flatten(),Z_hat_train[:,0].flatten())[0]
z2_train_corr = pearsonr(Z_train[:,1].flatten(),Z_hat_train[:,1].flatten())[0]

z1_test_corr = pearsonr(Z_test[:,0].flatten(),Z_hat_test[:,0].flatten())[0]
z2_test_corr = pearsonr(Z_test[:,1].flatten(),Z_hat_test[:,1].flatten())[0]

X_train_corr = pearsonr(train_spikes.flatten(),X_hat_train.detach().cpu().numpy().flatten())[0]
X_test_corr = pearsonr(test_spikes.flatten(),X_hat_test.detach().cpu().numpy().flatten())[0]

X_train_decoded_corr = pearsonr(train_spikes.flatten(),X_train_decoded.detach().cpu().numpy().flatten())[0]
X_test_decoded_corr = pearsonr(test_spikes.flatten(),X_test_decoded.detach().cpu().numpy().flatten())[0]




data_dict = {'z1_train_corr':z1_train_corr,
             'z1_test_corr':z1_test_corr,
             'z2_train_corr':z2_train_corr,
             'z2_test_corr':z2_test_corr,
             'X_train_corr':X_train_corr,
             'X_test_corr':X_test_corr,
             'X_train_decoded_corr':X_train_decoded_corr,
             'X_test_decoded_corr':X_test_decoded_corr}

print(data_dict)
np.save(f'figures/data/prob_{args.prob_of_measurement}_sample_{args.sample_number}.npy',data_dict)