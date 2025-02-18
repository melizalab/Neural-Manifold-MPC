import numpy as np
from torch.utils.data import DataLoader,TensorDataset
import torch
import pandas as pd

def concatenate_spikes(spike_data,measurement_indxs_path,prob,sample):
    # get measurement indexes
    indx_dict = pd.read_pickle(f'{measurement_indxs_path}.pkl').loc[(prob,sample)]

    # Get corresponding spikes
    sensory_spikes = spike_data['sensory_spikes'][indx_dict['sensory_indxs']]
    reservoir_spikes = spike_data['reservoir_spikes'][indx_dict['reservoir_indxs']]
    output_spikes = spike_data['output_spikes'][indx_dict['output_indxs']]

    # Concatenate spikes
    measured_spikes = np.concatenate((sensory_spikes, reservoir_spikes, output_spikes), axis=0).T
    return measured_spikes


def VAE_dataloader(measured_spikes,batch_size,shuffle=True):
    n_neurons = measured_spikes.shape[1]
    data_loader = DataLoader(torch.from_numpy(measured_spikes).type(torch.float32), batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return data_loader,n_neurons
