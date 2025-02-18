import numpy as np
import argparse
import torch
import sys

'''
# TODO:
Last time I changed V_assimilation to have train, val, and test splits.
Now need to make this work for the generate spikes so we will have
train, val, and test spikes.
'''

# ------------------------------
# Append path for custom modules
# ------------------------------
snn_path = r"C:\Users\chris\OneDrive\Desktop\MPC_of_Neural_Manifold"
sys.path.append(snn_path)
from network_architectures import CVAE
from network_architectures import rLIF_classification as nets
from latent_control.snn_scripts.SNN_diagnostics import clear_buffers

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--stimuli_VAE_model_path', default='latent_control/saved_models/stimulus_vaes/MNIST_VAE', type=str)
p.add_argument('--SNN_model_path',default='latent_control/saved_models/snns/SNN_classifier', type=str)
p.add_argument('--V_assimilation_in_path',default='latent_control/assimilation_data/V_assimilation', type=str)
p.add_argument('--spikes_assimilation_out_path',default='latent_control/assimilation_data/binary_assimilation_spikes', type=str)
args = p.parse_args()

# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')

# ------------------------
# Load V_assimilation Data
# ------------------------
print('Loading V_assimilation data...')
V_assimilation = np.load(f'{args.V_assimilation_in_path}.npy',allow_pickle=True)[()]

# --------------------------------------
# Load Stimulus VAE and Model Parameters
# --------------------------------------
print('Loading Stimulus VAE...')
CVAE_params = np.load(f'{args.stimuli_VAE_model_path}_parameters.npy',allow_pickle=True)[()]
stimulus_VAE = CVAE.VAE(CVAE_params).to(device)
stimulus_VAE.load_state_dict(torch.load(f'{args.stimuli_VAE_model_path}.pt'))
stimulus_VAE.eval()

# -----------------------------
# Load SNN and Model Parameters
# -----------------------------
print('LOADING SNN parameters...')
# Parameters
snn_params = np.load(f'{args.SNN_model_path}_parameters.npy', allow_pickle=True)[()]
snn_params['stimulus_n_steps'] = 1
snn_params['batch_size'] = 1
snn_params['is_online'] = True
# Weights
net_weights = torch.load(f'{args.SNN_model_path}.pt')
# Load
SNN = nets.rLIF(params=snn_params, return_out_V=True).to(device)
SNN.load_state_dict(clear_buffers(net_weights), strict=False)
SNN.eval()

# --------------------------------
# Stimulate SNN and Collect Spikes
# --------------------------------
X_split = ['X_train','X_test']
data_dict = {}
for split_indx,V_split in enumerate(V_assimilation.keys()):
    print(f'Stimulating SNN with {V_split} data...')
    ##############################
    # Import V assimilation data #
    ##############################
    V = torch.from_numpy(V_assimilation[V_split]).type(torch.float32)
    len_of_stim = V.shape[0]

    ########################
    # Stimulate SNN with V #
    ########################
    # Set up empty recorder for spikes
    sensory_spikes = np.zeros((snn_params['sensory_layer_size'], len_of_stim))
    reservoir_spikes = np.zeros((snn_params['reservoir_layer_size'], len_of_stim))
    output_spikes = np.zeros((10, len_of_stim))

    # Loop through stimuli images and collect spike trains
    with torch.no_grad():
        for time_step in range(len_of_stim):
            # Project latent V trajectory back into original stimulus dimension
            image = stimulus_VAE.decode(V[time_step].reshape(1,-1).to(device))
            # Use decoded image to stimulate SNN
            sense_spikes, res_spikes, out_spikes, out_V = SNN(image.reshape(1,784))
            sensory_spikes[:, time_step] = sense_spikes.cpu().detach().numpy().reshape(snn_params['sensory_layer_size'])
            reservoir_spikes[:, time_step] = res_spikes.cpu().detach().numpy().reshape(snn_params['reservoir_layer_size'])
            output_spikes[:, time_step] = out_spikes.cpu().detach().numpy().reshape(10)
            if time_step%1000 == 0:
                print(f'Time step: {time_step}/{len_of_stim}')
    data_dict[f'{X_split[split_indx]}']={'sensory_spikes':sensory_spikes,'reservoir_spikes':reservoir_spikes,'output_spikes':output_spikes}

# ----------------------
# Save Assimilation Data
# ----------------------
np.save(f'{args.spikes_assimilation_out_path}.npy',data_dict)
