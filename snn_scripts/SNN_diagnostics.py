import numpy as np
import matplotlib.pyplot as plt
import torch
import snntorch.functional as SF
from snntorch import utils

def predicted_digit(output_spikes,time_range):
    '''
    Takes output spike train and predicts digit based on which neuron has
    highest number of spikes
    :param output_spikes: 2D array of spike times (neuron indx x time step)
    :param time_range: 2 element array where first and last elements are the
                       lower and upper bounds of the time
                       window to calculate the predicted digit.
    :return: predicted digit
    '''
    spike_sums = np.sum(output_spikes[:,time_range[0]:time_range[1]],axis=1)
    prediction = np.argmax(spike_sums)
    return prediction

def plot_spikes(spikes, t, time_digit_switch=None, num_plot=None, t_range=None, labels=None, fontsize=12, tick_thickness=2, tick_length=6):
    if num_plot is None:
        num_plot = spikes.shape[0]
    if t_range is None:
        t_range = len(t)

    plt.rcParams.update({'font.size': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize,
                         'xtick.major.width': tick_thickness, 'ytick.major.width': tick_thickness,
                         'xtick.major.size': tick_length, 'ytick.major.size': tick_length})

    for i in range(num_plot):
        spk_t = t[np.where(spikes[i, :] == 1)]
        plt.vlines(spk_t, 0 + i, 0.8 + i, color='black')
        if time_digit_switch is not None:
            plt.vlines(time_digit_switch, 0, num_plot, linestyles='--', color='red', alpha=0.1)
        if labels is not None:
            for j, label in enumerate(labels):
                plt.hlines(label, time_digit_switch[j], time_digit_switch[j] + int(spikes.shape[1] / len(labels)))
    plt.xlim(time_digit_switch[0]-10,time_digit_switch[-1]+10)
    plt.show()

def clear_buffers(net_weights):
    filtered_net_weights = {k: v for k, v in net_weights.items() if not any(buf in k for buf in
                                                                            ['sensory_spikes', 'sensory_V',
                                                                             'reservoir_spikes', 'reservoir_V',
                                                                             'output_spikes', 'output_V'])}
    return filtered_net_weights



