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
p.add_argument('--path_to_mpc_data',type=str,default='./neural_manifold_control/mpc')
args = p.parse_args()

# Functions
def nMSE(z_ref,z_control):
    mse = np.mean((z_ref-z_control)**2)
    nmse = mse/(np.max(z_ref)-np.min(z_ref))
    return nmse

def get_arcs(ref_path,trial):
    arc_data = np.load(f'{args.path_to_mpc_data}/{ref_path}_control/prob_0.2_sample_0_trial_{trial}.npy',allow_pickle=True)[()]
    Z_control = arc_data['Z_control']
    Z_ref = arc_data['Z_ref']
    V = arc_data['V']
    return Z_control,Z_ref,V

# Plot params
n_trials = 50
arc_1_color='darkcyan'
arc_2_color='darkmagenta'
v1_color = '#1f77b4'
v2_color = '#ff7f0e'
unconst_color='darkred'
alpha = 0.05
linewidth=.1
fig, ax = plt.subplots(3, 2, figsize=(3,2.5), sharex=True, 
                       gridspec_kw={'height_ratios': [1, 1, 1]})  # Ensure uniform heights

# Share y-axis for rows 0 and 1 across both columns
ax[0,0].get_shared_y_axes().join(ax[0,0], ax[0,1])
ax[1,0].get_shared_y_axes().join(ax[1,0], ax[1,1])

# Share y-axis for columns in row 2
ax[2,0].get_shared_y_axes().join(ax[2,0], ax[2,1])


# Loop
arc_1_mean = np.zeros((1000,2))
arc_2_mean = np.zeros((1000,2))


for trial in range(n_trials):
    Z_control_arc_1,Z_ref_arc_1,V_arc_1 = get_arcs(ref_path='arc_1',trial=trial)
    Z_control_arc_2,Z_ref_arc_2,V_arc_2 = get_arcs(ref_path='arc_2',trial=trial)

    # Get mean controlled trajectory
    arc_1_mean += Z_control_arc_1
    arc_2_mean += Z_control_arc_2

    # Traj 1
    ax[0,0].plot(Z_control_arc_1[:,0],color=arc_1_color,alpha=alpha,linewidth=linewidth)
    ax[1,0].plot(Z_control_arc_1[:,1],color=arc_1_color,alpha=alpha,linewidth=linewidth)
    ax[2,0].plot(V_arc_1[:,0],alpha=alpha,linewidth=linewidth,color=v1_color)
    ax[2,0].plot(V_arc_1[:,1],alpha=alpha,linewidth=linewidth,color=v2_color)
    


    # Traj 2
    ax[0,1].plot(Z_control_arc_2[:,0],color=arc_2_color,alpha=alpha,linewidth=linewidth)
    ax[1,1].plot(Z_control_arc_2[:,1],color=arc_2_color,alpha=alpha,linewidth=linewidth)
    ax[2,1].plot(V_arc_2[:,0],alpha=alpha,linewidth=linewidth,color=v1_color)
    ax[2,1].plot(V_arc_2[:,1],alpha=alpha,linewidth=linewidth,color=v2_color)


arc_1_mean/=n_trials
arc_2_mean/=n_trials

ax[0,0].plot(arc_1_mean[:,0],color=arc_1_color,alpha=0.8,linewidth=linewidth*3)
ax[1,0].plot(arc_1_mean[:,1],color=arc_1_color,alpha=0.8,linewidth=linewidth*3)

ax[0,1].plot(arc_2_mean[:,0],color=arc_2_color,alpha=0.8,linewidth=linewidth*3)
ax[1,1].plot(arc_2_mean[:,1],color=arc_2_color,alpha=0.8,linewidth=linewidth*3)

ax[0,0].plot(Z_ref_arc_1[:,0],color='black',alpha=0.5,linewidth=1)
ax[0,1].plot(Z_ref_arc_1[:,1],color='black',alpha=0.5,linewidth=1)
ax[1,0].plot(Z_ref_arc_2[:,0],color='black',alpha=0.5,linewidth=1)
ax[1,1].plot(Z_ref_arc_2[:,1],color='black',alpha=0.5,linewidth=1)

ax[0,0].set_xticks([0,250,500,750,1000])
ax[0,0].set_ylim([-0.07,0.07])
ax[1,0].set_ylim([-0.07,0.07])

ax[0,1].set_yticklabels([])
ax[1,1].set_yticklabels([])
ax[2,1].set_yticklabels([])
plt.savefig('figures/raw_figures/figure_6_B_latent_trajectories.pdf')

