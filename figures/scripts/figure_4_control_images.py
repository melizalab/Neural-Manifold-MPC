import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
from network_architectures.latent_linear_dynamics import LDM


# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_p_data',type=str,default='./neural_manifold_control/reactive_control/p_control/set_point_control')
p.add_argument('--path_to_mpc_data',type=str,default='./neural_manifold_control/mpc/set_point_control')
p.add_argument('--path_to_LDM',default='saved_models/latent_dynamics_models/LDM_prob_0.2_sample_0')
args = p.parse_args()

# Load control data
p_data = np.load(f'{args.path_to_p_data}/prob_0.2_sample_0_trial_0.npy',allow_pickle=True)[()]
mpc_data = np.load(f'{args.path_to_mpc_data}/prob_0.2_sample_0_trial_0.npy',allow_pickle=True)[()]

# Load LDM
# ---------
# Load LDM
# ---------
print('Loading LDM weights...')
ldm_dict = torch.load(f'{args.path_to_LDM}.pt')
ldm = LDM(ldm_dict["snn_params"]['snn_vae_model_path'],ldm_dict["mnist_params"]['mnist_cvae_model_path'])
ldm.load_state_dict(ldm_dict['model_state_dict'])
ldm.eval()

times = [200,500,800]

fig,ax = plt.subplots(2,3,figsize=(6,3))
for i,time in enumerate(times):
    p_u = ldm.u_decoder.decode(torch.from_numpy(p_data['V'][time]).type(torch.float32).reshape(1,-1))
    mpc_u = ldm.u_decoder.decode(torch.from_numpy(mpc_data['V'][time]).type(torch.float32).reshape(1,-1))

    ax[0,i].imshow(p_u.detach().cpu().reshape((28,28)))
    ax[1,i].imshow(mpc_u.detach().cpu().reshape((28,28)))

ax[0,0].set_title('200 ms')
ax[0,1].set_title('500 ms')
ax[0,2].set_title('800 ms')
ax[0,0].set_ylabel('Prop. Control')
ax[1,0].set_ylabel('MPC')
for a in ax.flatten():
    a.set_xticks([])
    a.set_yticks([])
plt.savefig('figures/raw_figures/figure_4_control_images.pdf')