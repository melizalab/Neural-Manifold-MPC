import numpy as np
import matplotlib.pyplot as plt
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

n_probs = 8
n_samples = 10
n_trials = 50

mpc_nMSE_z1 = np.zeros((n_probs,n_samples,n_trials))
mpc_nMSE_z2 = np.zeros_like(mpc_nMSE_z1)
p_nMSE_z1 = np.zeros_like(mpc_nMSE_z1)
p_nMSE_z2 = np.zeros_like(mpc_nMSE_z1)


p_color='darkgoldenrod'
mpc_color='darkred'
for i,prob in enumerate([0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6]):
    for sample in range(n_samples):
        for trial in range(n_trials):
            p_file = f'{args.path_to_p_data}/prob_{prob}_sample_{sample}_trial_{trial}.npy'
            mpc_file = f'{args.path_to_mpc_data}/prob_{prob}_sample_{sample}_trial_{trial}.npy'

            p_data = np.load(p_file,allow_pickle=True)[()]
            mpc_data = np.load(mpc_file,allow_pickle=True)[()]

            # Get ref traj
            Z_ref = mpc_data['Z_ref']

            # Get nMSEs
            p_nMSE_z1[i,sample,trial] = nMSE(Z_ref[:,0],p_data['Z_control'][:,0])
            p_nMSE_z2[i,sample,trial] = nMSE(Z_ref[:,1],p_data['Z_control'][:,1])

            mpc_nMSE_z1[i,sample,trial] = nMSE(Z_ref[:,0],mpc_data['Z_control'][:,0])
            mpc_nMSE_z2[i,sample,trial] = nMSE(Z_ref[:,1],mpc_data['Z_control'][:,1])


p_nMSE = (p_nMSE_z1+p_nMSE_z2)/2
mpc_nMSE= (mpc_nMSE_z1+mpc_nMSE_z2)/2
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

colors = ['darkgoldenrod', 'darkred']

fig, ax = plt.subplots(figsize=(2,2))
probs = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6]
for i,prob in enumerate(probs):
    positions = np.array([-0.20,0.20])+i*2
    data = [p_nMSE[i].mean(0),mpc_nMSE[i].mean(0)]
    
    # Create boxplot
    box = ax.boxplot(
        data, 
        positions=positions, 
        widths=[0.3*1.25, 0.3*1.25],
        patch_artist=True,  # Fill the boxes
        medianprops={'color': 'black', 'linewidth': 1.5},  
        whiskerprops={'linewidth': 1.2},  
        capprops={'linewidth': 1.2},  
        flierprops={'marker': 'o', 'markersize': 2, 'markeredgewidth': 1},  
    )

    # Apply colors and styles
    for j, patch in enumerate(box['boxes']):
        if j == 0:  # P-control
            patch.set_facecolor(p_color)
            patch.set_alpha(0.6)
        else:  # MPC
            patch.set_facecolor(mpc_color)
            patch.set_alpha(0.7)

    # Assign correct colors to whiskers & caps
    for j in range(0, len(box['whiskers']), 2):  # Whiskers are in pairs (left and right)
        color = p_color if j < len(box['whiskers']) // 2 else mpc_color
        box['whiskers'][j].set_color(color)
        box['whiskers'][j + 1].set_color(color)

    for j in range(0, len(box['caps']), 2):  # Caps are also in pairs
        color = p_color if j < len(box['caps']) // 2 else mpc_color
        box['caps'][j].set_color(color)
        box['caps'][j + 1].set_color(color)

    # Apply matching colors to outliers
    for j, flier in enumerate(box['fliers']):
        flier.set_markeredgecolor(p_color if j % 2 == 0 else mpc_color)
        flier.set_marker('o' if j % 2 == 0 else 'D')  # Circle for P, Diamond for MPC

# Customize X-axis
ax.set_xticks(np.arange(0, 2 * n_probs, 2))
ax.set_xticklabels([f'{p}' for p in probs])
ax.set_xlabel("Sample Probability")
ax.set_ylabel("nMSE")

#plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig('figures/raw_figures/figure_5_B_p_vs_mpc_latent_nMSEs.pdf')