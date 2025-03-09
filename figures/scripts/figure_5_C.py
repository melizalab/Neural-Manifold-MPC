import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 6,     # X-axis tick labels
    'ytick.labelsize': 6,     # Y-axis tick labels
})
# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_p_data',type=str,default='./neural_manifold_control/reactive_control/p_control/set_point_control')
p.add_argument('--path_to_mpc_data',type=str,default='./neural_manifold_control/mpc/set_point_control')
args = p.parse_args()

def nMSE(z_ref,z_control):
    mse = np.mean((z_ref-z_control)**2)
    nmse = mse/(np.max(z_ref)-np.min(z_ref))
    return nmse


linewidth = 1
n_trials = 50
p_color='darkgoldenrod'
mpc_color='darkred'
alpha = 0.1
linewidth = .5
'''

Get best trials
prob = 0.01
p_nMSE[0].mean(1).argmin() = 1

prob = 0.05
p_nMSE[1].mean(1).argmin() = 4

prob = 0.1
p_nMSE[2].mean(1).argmin() = 1

prob = 0.2
p_nMSE[3].mean(1).argmin() = 0

prob = 0.3
p_nMSE[4].mean(1).argmin() = 2

prob = 0.4
p_nMSE[5].mean(1).argmin() = 4

prob = 0.5
p_nMSE[6].mean(1).argmin() = 2

prob = 0.6
p_nMSE[7].mean(1).argmin() = 9
'''

import matplotlib.pyplot as plt
import numpy as np

probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
best_samples = [1, 4, 1, 0, 2, 4, 2, 9]




probs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6]
best_samples = [1,4,1,0,2,4,2,9]
fig,ax = plt.subplots(4,4,sharex=True,sharey=True,figsize=(5,4))
ylims =[]
for i, prob in enumerate(probs):
    for trial in range(n_trials):
        p_file = f'{args.path_to_p_data}/prob_{prob}_sample_{best_samples[i]}_trial_{trial}.npy'
        mpc_file = f'{args.path_to_mpc_data}/prob_{prob}_sample_{best_samples[i]}_trial_{trial}.npy'

        p_data = np.load(p_file, allow_pickle=True)[()]
        mpc_data = np.load(mpc_file, allow_pickle=True)[()]

        # Get reference trajectory
        Z_ref = mpc_data['Z_ref']
        Z1_ref_min, Z1_ref_max = Z_ref[:, 0].min(), Z_ref[:, 0].max()
        Z2_ref_min, Z2_ref_max = Z_ref[:, 1].min(), Z_ref[:, 1].max()

        # Normalize all data relative to its own Z_ref scale
        Z1_ref_norm = (Z_ref[:, 0] - Z1_ref_min) / (Z1_ref_max - Z1_ref_min)
        Z2_ref_norm = (Z_ref[:, 1] - Z2_ref_min) / (Z2_ref_max - Z2_ref_min)

        Z1_p_norm = (p_data['Z_control'][:, 0] - Z1_ref_min) / (Z1_ref_max - Z1_ref_min)
        Z2_p_norm = (p_data['Z_control'][:, 1] - Z2_ref_min) / (Z2_ref_max - Z2_ref_min)

        Z1_mpc_norm = (mpc_data['Z_control'][:, 0] - Z1_ref_min) / (Z1_ref_max - Z1_ref_min)
        Z2_mpc_norm = (mpc_data['Z_control'][:, 1] - Z2_ref_min) / (Z2_ref_max - Z2_ref_min)

        # Plot in the correct subplot
        if i < 4:
            ax[0, i].plot(Z1_ref_norm, color='black', alpha=0.5, linewidth=1)
            ax[1, i].plot(Z2_ref_norm, color='black', alpha=0.5, linewidth=1)

            ax[0, i].plot(Z1_p_norm, color=p_color, alpha=alpha, linewidth=linewidth)
            ax[1, i].plot(Z2_p_norm, color=p_color, alpha=alpha, linewidth=linewidth)

            ax[0, i].plot(Z1_mpc_norm, color=mpc_color, alpha=alpha, linewidth=linewidth)
            ax[1, i].plot(Z2_mpc_norm, color=mpc_color, alpha=alpha, linewidth=linewidth)

        else:
            ax[2, i-4].plot(Z1_ref_norm, color='black', alpha=0.5, linewidth=1)
            ax[3, i-4].plot(Z2_ref_norm, color='black', alpha=0.5, linewidth=1)

            ax[2, i-4].plot(Z1_p_norm, color=p_color, alpha=alpha, linewidth=linewidth)
            ax[3, i-4].plot(Z2_p_norm, color=p_color, alpha=alpha, linewidth=linewidth)

            ax[2, i-4].plot(Z1_mpc_norm, color=mpc_color, alpha=alpha, linewidth=linewidth)
            ax[3, i-4].plot(Z2_mpc_norm, color=mpc_color, alpha=alpha, linewidth=linewidth)
for j,a in enumerate(ax.flatten()):
    a.set_xticks([0,250,500,750,1000])
    a.set_ylim([-3,3])


plt.tight_layout()
plt.savefig('figures/raw_figures/figure_5_C_latent_trajectories.pdf')