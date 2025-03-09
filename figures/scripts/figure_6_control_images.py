import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from network_architectures.latent_linear_dynamics import LDM
plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 7,     # X-axis tick labels
    'ytick.labelsize': 7,     # Y-axis tick labels
})

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_data',type=str,default='./neural_manifold_control/mpc')
p.add_argument('--path_to_LDM',default='saved_models/latent_dynamics_models/LDM_prob_0.2_sample_0')
args = p.parse_args()

# Load control data
arc_1_data = np.load(f'{args.path_to_data}/arc_1_control/prob_0.2_sample_0_trial_0.npy',allow_pickle=True)[()]
arc_2_data = np.load(f'{args.path_to_data}/arc_2_control/prob_0.2_sample_0_trial_0.npy',allow_pickle=True)[()]


# ---------
# Load LDM
# ---------
print('Loading LDM weights...')
ldm_dict = torch.load(f'{args.path_to_LDM}.pt')
ldm = LDM(ldm_dict["snn_params"]['snn_vae_model_path'],ldm_dict["mnist_params"]['mnist_cvae_model_path'])
ldm.load_state_dict(ldm_dict['model_state_dict'])
ldm.eval()

times = [0,250,500,750,999]

fig,ax = plt.subplots(2,5,figsize=(3.5,2))
for i,time in enumerate(times):
    arc_1_u = ldm.u_decoder.decode(torch.from_numpy(arc_1_data['V'][time]).type(torch.float32).reshape(1,-1))
    arc_2_u = ldm.u_decoder.decode(torch.from_numpy(arc_2_data['V'][time]).type(torch.float32).reshape(1,-1))

    ax[0,i].imshow(arc_1_u.detach().cpu().reshape((28,28)))
    ax[1,i].imshow(arc_2_u.detach().cpu().reshape((28,28)))

ax[0,0].set_title('0 ms')
ax[0,1].set_title('250 ms')
ax[0,2].set_title('500 ms')
ax[0,3].set_title('750 ms')
ax[0,4].set_title('999 ms')

ax[0,0].set_ylabel('Traj 1')
ax[1,0].set_ylabel('Traj 2')
for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])
plt.savefig('figures/raw_figures/figure_7_B_control_images.pdf')