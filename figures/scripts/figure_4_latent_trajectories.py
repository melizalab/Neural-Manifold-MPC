import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse

plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 7,     # X-axis tick labels
    'ytick.labelsize': 7,     # Y-axis tick labels
})
# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_p_data',type=str,default='./neural_manifold_control/reactive_control/pid_control/set_point_control')
p.add_argument('--path_to_mpc_data',type=str,default='./neural_manifold_control/mpc/set_point_control')
args = p.parse_args()

def nMSE(z_ref,z_control):
    mse = np.mean((z_ref-z_control)**2)
    nmse = mse/(np.max(z_ref)-np.min(z_ref))
    return nmse

fig,ax = plt.subplots(4,1,sharex=True,figsize=(6,6))



mpc_nMSE_z1 = 0
mpc_nMSE_z2 = 0
p_nMSE_z1 = 0
p_nMSE_z2 = 0

# Loop through files
num_files = 0
linewidth = 1.5
alpha = 0.1
n_trials = 10
p_color='darkgoldenrod'
mpc_color='darkred'

for trial in range(n_trials):
    p_file = f'{args.path_to_p_data}/prob_0.2_sample_0_trial_{trial}.npy'
    mpc_file = f'{args.path_to_mpc_data}/prob_0.2_sample_0_trial_{trial}.npy'

    p_data = np.load(p_file,allow_pickle=True)[()]
    mpc_data = np.load(mpc_file,allow_pickle=True)[()]

    # Get ref traj
    Z_ref = mpc_data['Z_ref']

    # Get nMSEs
    p_nMSE_z1 += nMSE(Z_ref[:,0],p_data['Z_control'][:,0])
    p_nMSE_z2 += nMSE(Z_ref[:,1],p_data['Z_control'][:,1])

    mpc_nMSE_z1 += nMSE(Z_ref[:,0],mpc_data['Z_control'][:,0])
    mpc_nMSE_z2 += nMSE(Z_ref[:,1],mpc_data['Z_control'][:,1])

    # plot p data
    # Z states
    ax[0].plot(p_data['Z_control'][:,0],color=p_color,alpha=alpha,linewidth=linewidth)
    ax[1].plot(p_data['Z_control'][:,1],color=p_color,alpha=alpha,linewidth=linewidth)
    # V Inputs
    ax[2].plot(p_data['V'][:,0],color=p_color,alpha=alpha,linewidth=linewidth)
    ax[3].plot(p_data['V'][:,1],color=p_color,alpha=alpha,linewidth=linewidth)

    # plot mpc data
    # Z states
    ax[0].plot(mpc_data['Z_control'][:,0],color=mpc_color,alpha=alpha,linewidth=linewidth)
    ax[1].plot(mpc_data['Z_control'][:,1],color=mpc_color,alpha=alpha,linewidth=linewidth)
    # V Inputs
    ax[2].plot(mpc_data['V'][:,0],color=mpc_color,alpha=alpha,linewidth=linewidth)
    ax[3].plot(mpc_data['V'][:,1],color=mpc_color,alpha=alpha,linewidth=linewidth)
# plot ref
ax[0].plot(Z_ref[:,0],color='black',alpha=0.5,linewidth=1)
ax[1].plot(Z_ref[:,1],color='black',alpha=0.5,linewidth=1)

ax[0].set_ylabel('Z1')
ax[1].set_ylabel('Z2')
ax[2].set_ylabel('V1')
ax[3].set_ylabel('V2')
ax[3].set_xlabel('Time (ms)')
ax[3].set_xticks([0,250,500,750,1000])



ax[0].set_ylim([-0.15,0.15])
ax[1].set_ylim([-0.15,0.15])

ax[2].set_ylim([-2,2])
ax[3].set_ylim([-0.8,0.8])
ax[3].set_xlim([0,1000])

# Get mean and std of nMSE
p_nMSE_z1/=n_trials
p_nMSE_z2/=n_trials
mpc_nMSE_z1/=n_trials
mpc_nMSE_z2/=n_trials

print(f'P-Control nMSEs: Z1 = {p_nMSE_z1}, Z2 = {p_nMSE_z2} ')
print(f'MPC nMSEs: Z1 = {mpc_nMSE_z1}, Z2 = {mpc_nMSE_z2} ')
#plt.show()
plt.savefig('figures/pid_figures/figure_4_latent_trajectories.pdf')
