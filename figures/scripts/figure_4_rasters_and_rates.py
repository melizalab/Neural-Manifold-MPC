import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import gaussian_filter1d


plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 7,     # X-axis tick labels
    'ytick.labelsize': 7,     # Y-axis tick labels
})
p_color='darkgoldenrod'
mpc_color='darkred'
alpha= 0.5
linewidths =1
# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_p_data',type=str,default='./neural_manifold_control/reactive_control/p_control/set_point_control')
p.add_argument('--path_to_mpc_data',type=str,default='./neural_manifold_control/mpc/set_point_control')
args = p.parse_args()

# Load control data
p_data = np.load(f'{args.path_to_p_data}/prob_0.2_sample_0_trial_0.npy',allow_pickle=True)[()]
mpc_data = np.load(f'{args.path_to_mpc_data}/prob_0.2_sample_0_trial_0.npy',allow_pickle=True)[()]


p_spikes = np.zeros((1000))
mpc_spikes = np.zeros((1000))

for trial in range(50):
    p_file = f'{args.path_to_p_data}/prob_0.2_sample_0_trial_{trial}.npy'
    mpc_file = f'{args.path_to_mpc_data}/prob_0.2_sample_0_trial_{trial}.npy'

    p_data = np.load(p_file,allow_pickle=True)[()]
    mpc_data = np.load(mpc_file,allow_pickle=True)[()]

    p_spikes += p_data['spikes'].sum(1)
    mpc_spikes += mpc_data['spikes'].sum(1)

# Apply Gaussian smoothing
sigma = 5  # Adjust smoothing level (higher = more smoothing)
p_spikes_smooth = gaussian_filter1d(p_spikes/50, sigma=sigma)
mpc_spikes_smooth = gaussian_filter1d(mpc_spikes/50, sigma=sigma)
fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(6,1))

ax[0].plot(p_spikes_smooth,color='black',alpha=alpha,linewidth=linewidths)
ax[0].plot(gaussian_filter1d(p_data['spikes'].sum(1), sigma=sigma),color=p_color,alpha=alpha,linewidth=linewidths)
ax[1].plot(mpc_spikes_smooth,color='black',alpha=alpha,linewidth=linewidths)
ax[1].plot(gaussian_filter1d(mpc_data['spikes'].sum(1), sigma=sigma),color=mpc_color,alpha=alpha,linewidth=linewidths)

ax[1].set_xticks([0,250,500,750,1000])
ax[1].set_xlim([0,1000])
ax[1].set_yticks([2,4,6])
plt.savefig('figures/raw_figures/figure_4_rates.pdf')

fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(6,2))
t = np.arange(0, 1000)  # Time vector

# Iterate over each neuron (assumed to be 122 neurons)
for i in range(122):
    # Get spike times for the p_data and mpc_data for each neuron
    p_spike_times = np.where(p_data['spikes'][:, i] > 0)[0]  # Indices where spikes occur
    mpc_spike_times = np.where(mpc_data['spikes'][:, i] > 0)[0]  # Indices where spikes occur
    
    # Plot vertical lines (spikes) for p_data and mpc_data
    ax[0].vlines(t[p_spike_times], 0 + i, 0.8 + i, color='black', linewidth=0.3)
    ax[1].vlines(t[mpc_spike_times], 0 + i, 0.8 + i, color='black', linewidth=0.3)

# Set x-ticks and limits
ax[1].set_xticks([0, 250, 500, 750, 1000])
ax[1].set_xlim([0, 1000])

plt.savefig('figures/raw_figures/figure_4_rasters.pdf')
