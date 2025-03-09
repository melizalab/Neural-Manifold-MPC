import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse
import torch

plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 7,     # X-axis tick labels
    'ytick.labelsize': 7,     # Y-axis tick labels
})
# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_p_data',type=str,default='./neural_manifold_control/reactive_control/p_control/set_point_control/prob_0.2_sample_0_trial_0')
p.add_argument('--path_to_mpc_data',type=str,default='./neural_manifold_control/mpc/set_point_control/prob_0.2_sample_0_trial_0')
p.add_argument('--path_to_LDM',default='saved_models/latent_dynamics_models/LDM_prob_0.2_sample_0')
args = p.parse_args()

# Load control data
p_data = np.load(f'{args.path_to_p_data}.npy',allow_pickle=True)[()]
mpc_data = np.load(f'{args.path_to_mpc_data}.npy',allow_pickle=True)[()]

# Load coefficients
ldm_dict = torch.load(f'{args.path_to_LDM}.pt')
AB_weights = ldm_dict['model_state_dict']['AB_dynamics.weight'].cpu().numpy()
A = AB_weights[:,:2]
B = AB_weights[:,2:]


p_color='darkgoldenrod'
mpc_color='darkred'


def get_vector_field(A,B,v):
    # Define grid range for state space
    z1_range = np.linspace(-.15, .15, 10)
    z2_range = np.linspace(-.15, .15, 10)

    Z1, Z2 = np.meshgrid(z1_range, z2_range)

    # Compute vector field
    dZ1 = np.zeros_like(Z1)
    dZ2 = np.zeros_like(Z2)

    for i in range(Z1.shape[0]):
        for j in range(Z1.shape[1]):
            z = np.array([Z1[i, j], Z2[i, j]])
            dz = (A - np.eye(2)) @ z + B @ v  # Compute Δz
            dZ1[i, j], dZ2[i, j] = dz[0], dz[1]
    return dZ1,dZ2,Z1,Z2


def get_zero(A,B,v):
    return np.linalg.inv((np.eye(2)-A))@B@v


fig,ax = plt.subplots(1,3,sharex=True,sharey=True,figsize=(6,2))
alpha = 0.5
for i,time in enumerate([200,500,800]):
    p_dZ1,p_dZ2,Z1,Z2 = get_vector_field(A,B,p_data['V'][time,:])
    mpc_dZ1,mpc_dZ2,_,_ = get_vector_field(A,B,mpc_data['V'][time,:])

    # plot latent state
    mpc_state = mpc_data['Z_control'][time,:]
    p_state = p_data['Z_control'][time,:]
    ax[i].scatter(p_state[0],p_state[1],s=10,color=p_color)
    ax[i].scatter(mpc_state[0],mpc_state[1],s=10,color=mpc_color)

    # plot reference point
    ax[i].scatter(mpc_data['Z_ref'][time,0],mpc_data['Z_ref'][time,0],s=20,color='black',marker='x')

    # plot vector fields
    ax[i].quiver(Z1, Z2, p_dZ1, p_dZ2, angles="xy", scale_units="xy", scale=7, color=p_color,alpha=alpha)
    ax[i].quiver(Z1, Z2, mpc_dZ1, mpc_dZ2, angles="xy", scale_units="xy", scale=7, color=mpc_color,alpha=alpha-.1)
ax[0].set_yticks([-0.1,0,0.1])
ax[1].set_xticks([-0.1,0,0.1])

ax[0].set_title('200 ms')
ax[1].set_title('500 ms')
ax[2].set_title('800 ms')

plt.savefig('figures/raw_figures/figure_4_vector_field.pdf')

