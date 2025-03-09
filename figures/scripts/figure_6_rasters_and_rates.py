import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.ndimage import gaussian_filter1d
plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 6,     # X-axis tick labels
    'ytick.labelsize': 6,     # Y-axis tick labels
})
# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_mpc_data',type=str,default='./neural_manifold_control/mpc')
args = p.parse_args()


# Plot params
n_trials = 50
arc_1_color='darkcyan'
arc_2_color='darkmagenta'
v1_color = '#1f77b4'
v2_color = '#ff7f0e'
alpha = 0.05
linewidth=.1

# -----------
# Get Rasters
# -----------
spikes_arc_1 = np.load(f'{args.path_to_mpc_data}/arc_1_control/prob_0.2_sample_0_trial_0.npy',allow_pickle=True)[()]['spikes']
spikes_arc_2 = np.load(f'{args.path_to_mpc_data}/arc_2_control/prob_0.2_sample_0_trial_0.npy',allow_pickle=True)[()]['spikes']

fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(4,2))

t = np.arange(0,1000)
for i in range(122):
    arc_1_spike_times = np.where(spikes_arc_1[:,i] > 0)[0]
    arc_2_spike_times = np.where(spikes_arc_2[:,i]>0)[0]
    # Plot vertical lines (spikes) for p_data and mpc_data
    ax[0].vlines(t[arc_1_spike_times], 0 + i, 0.8 + i, color='black', linewidth=0.3)
    ax[1].vlines(t[arc_2_spike_times], 0 + i, 0.8 + i, color='black', linewidth=0.3)
ax[1].set_xlabel('Time (ms)')
ax[1].set_xticks([0,250,500,750,1000])
plt.tight_layout()
plt.savefig('figures/raw_figures/figure_7_B_rasters.pdf')
plt.close()
# ---------
# Get Rates
# ---------
sigma = 5
arc_1_rates = np.zeros((1000))
arc_2_rates = np.zeros((1000))
n_trials = 50
for trial in range(n_trials):
    
    spikes_arc_1 = np.load(f'{args.path_to_mpc_data}/arc_1_control/prob_0.2_sample_0_trial_{trial}.npy',allow_pickle=True)[()]['spikes']
    spikes_arc_2 = np.load(f'{args.path_to_mpc_data}/arc_2_control/prob_0.2_sample_0_trial_{trial}.npy',allow_pickle=True)[()]['spikes']
    if trial == 0:
        arc_1_trial_0_rate = gaussian_filter1d(spikes_arc_1.sum(1), sigma=sigma)
        arc_2_trial_0_rate = gaussian_filter1d(spikes_arc_2.sum(1), sigma=sigma)

    arc_1_rates += spikes_arc_1.sum(1)
    arc_2_rates += spikes_arc_2.sum(1)


arc_1_spikes_smooth = gaussian_filter1d(arc_1_rates/n_trials, sigma=sigma)
arc_2_spikes_smooth = gaussian_filter1d(arc_2_rates/n_trials, sigma=sigma)

fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(3.5,1))
ax[0].plot(arc_1_trial_0_rate,color='darkred',alpha=0.5)
ax[0].plot(arc_1_spikes_smooth,color='black',alpha=0.5)

ax[1].plot(arc_2_trial_0_rate,color='darkred',alpha=0.5)
ax[1].plot(arc_2_spikes_smooth,color='black',alpha=0.5)
plt.savefig('figures/raw_figures/figure_7_B_rates.pdf')
