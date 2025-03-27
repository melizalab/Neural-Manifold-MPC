import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 6,     # X-axis tick labels
    'ytick.labelsize': 6,     # Y-axis tick labels
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

def RMS(V):
    return np.sqrt(np.mean(V ** 2))

linewidth = 1
n_trials = 50
p_color='darkgoldenrod'
mpc_color='darkred'
alpha = 0.1
linewidth = .5

probs = [0.01, 0.1, 0.3, 0.6]
prob_exemplar_dict = {
    0.01: 1,
    0.1: 1,
    0.3: 4,
    0.6: 6
}



fig,ax = plt.subplots(4,4,sharex=True,figsize=(5,4))

for i,prob in enumerate(probs):
    v1_pid_max = -1000
    v1_pid_min = 1000
    v2_pid_max = -1000
    v2_pid_min = 1000
    for j,trial in enumerate(range(n_trials)):
        p_file = f'{args.path_to_p_data}/prob_{prob}_sample_{prob_exemplar_dict[prob]}_trial_{trial}.npy'
        mpc_file = f'{args.path_to_mpc_data}/prob_{prob}_sample_{prob_exemplar_dict[prob]}_trial_{trial}.npy'

        p_data = np.load(p_file,allow_pickle=True)[()]
        mpc_data = np.load(mpc_file,allow_pickle=True)[()]


        # Get ref traj
        if j == 0: # only need to do once per trial
            Z_ref = mpc_data['Z_ref']
            Z1_ref_range = np.abs(Z_ref[0,0]-Z_ref[-1,0])
            Z2_ref_range = np.abs(Z_ref[0,1]-Z_ref[-1,1])
            Z1_ref_range*=2
            Z2_ref_range*=2
            ax[0,i].plot(Z_ref[:,0],color='black',alpha=0.5,zorder=3)
            ax[1,i].plot(Z_ref[:,1],color='black',alpha=0.5,zorder=3)

        # Get Z data
        z_pid = p_data['Z_control']
        z_mpc = mpc_data['Z_control']
        # Get V data
        v_pid = p_data['V']
        v_mpc = mpc_data['V']

        z_range_scaling = 1.25
        # Plot Z1
        ax[0,i].plot(z_pid[:,0],alpha=alpha,linewidth=linewidth,color=p_color)
        ax[0,i].plot(z_mpc[:,0],alpha=alpha,linewidth=linewidth,color=mpc_color)
        ax[0,i].set_ylim([Z_ref[:,0].min()-Z1_ref_range*z_range_scaling,Z_ref[:,0].max()+Z1_ref_range*z_range_scaling])
        ax[0,i].set_yticks([Z_ref[:,0].min()-Z1_ref_range,0,Z_ref[:,0].max()+Z1_ref_range])

        # Plot Z2
        ax[1,i].plot(z_pid[:,1],alpha=alpha,linewidth=linewidth,color=p_color)
        ax[1,i].plot(z_mpc[:,1],alpha=alpha,linewidth=linewidth,color=mpc_color)
        ax[1,i].set_ylim([Z_ref[:,1].min()-Z2_ref_range*z_range_scaling,Z_ref[:,1].max()+Z2_ref_range*z_range_scaling])
        ax[1,i].set_yticks([Z_ref[:,1].min()-Z2_ref_range,0,Z_ref[:,1].max()+Z2_ref_range])

        # Standardize by v_mpc
        v1_pid_max = max(v1_pid_max, v_pid[:,0].max())
        v1_pid_min = min(v1_pid_min, v_pid[:,0].min())
        v2_pid_max = max(v2_pid_max, v_pid[:,1].max())
        v2_pid_min = min(v2_pid_min, v_pid[:,1].min())

        # Plot V1
        ax[2,i].plot(v_pid[:,0],alpha=alpha,linewidth=linewidth,color=p_color)
        ax[2,i].plot(v_mpc[:,0],alpha=alpha,linewidth=linewidth,color=mpc_color)

        # Plot V2
        ax[3,i].plot(v_pid[:,1],alpha=alpha,linewidth=linewidth,color=p_color)
        ax[3,i].plot(v_mpc[:,1],alpha=alpha,linewidth=linewidth,color=mpc_color)

        # Define number of ticks you want
        num_ticks = 3

        # Calculate range for V1 and V2
        V1_range = v1_pid_max - v1_pid_min
        V2_range = v2_pid_max - v2_pid_min

        # Define a little margin to ensure ticks don't cut off at the edges
        v_range_scaling = 2

        # Set the y-limits
        ax[2,i].set_ylim(v1_pid_min - V1_range * v_range_scaling, v1_pid_max + V1_range * v_range_scaling)
        ax[3,i].set_ylim(v2_pid_min - V2_range * v_range_scaling, v2_pid_max + V2_range * v_range_scaling)

        # Use np.linspace to generate evenly spaced ticks
        v1_ticks = np.linspace(v1_pid_min - V1_range, v1_pid_max + V1_range, num=num_ticks)
        v2_ticks = np.linspace(v2_pid_min - V2_range, v2_pid_max + V2_range, num=num_ticks)

        # Set the yticks
        ax[2,i].set_yticks(v1_ticks)
        ax[3,i].set_yticks(v2_ticks)


for j,a in enumerate(ax.flatten()):
    a.set_xticks([0,250,500,750,1000])
for i in range(4):
    ax[0,i].set_yticklabels([-2,0,2])
    ax[1,i].set_yticklabels([-2,0,2])
    ax[2,i].set_yticklabels([-1,0,1])
    ax[3,i].set_yticklabels([-1,0,1])

plt.tight_layout()
plt.savefig('figures/pid_figures/figure_5_B_and_C.pdf')



