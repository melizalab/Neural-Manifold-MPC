import argparse
import do_mpc
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
import pandas as pd
from tqdm import tqdm

from snn_scripts.SNN_diagnostics import clear_buffers
from network_architectures import rLIF_classification as SNN
from network_architectures.latent_linear_dynamics import LDM

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_SNN',type=str,default='saved_models/snns/SNN_classifier')
p.add_argument('--path_to_LDM',type=str,default='saved_models/latent_dynamics_models/LDM_prob_0.2_sample_0')
p.add_argument('--path_to_reference_trajectory',type=str,default='reference_trajectories/set_points')
p.add_argument('--trial_id',type=int,default=0)
p.add_argument('--path_to_save_output',type=str,default='neural_manifold_control/reactive_control/p_control/set_point_control')
p.add_argument('--arc_num',type=int,default=1)
p.add_argument('--p_gains',nargs='+',default=[90,0.5],type=float)
args = p.parse_args()

# ----------
# Import SNN
# ----------

print('Loading SNN...')
snn_dict = torch.load(f'{args.path_to_SNN}.pt')
snn_params = snn_dict['model_params']
snn_params['is_online'] = True
snn_params['stimulus_n_steps'] = 1
snn_params['batch_size'] = 1
snn = SNN.rLIF(params=snn_params, return_out_V=True)
snn.load_state_dict(clear_buffers(snn_dict['model_state_dict']), strict=False)
snn.eval()

# ----------
# Import LDM
# ----------
print('Loading LDM parameters...')
print('LOADING LDM weights and parameters...')
ldm_dict = torch.load(f'{args.path_to_LDM}.pt')
ldm = LDM(ldm_dict["snn_params"]['snn_vae_model_path'],ldm_dict["mnist_params"]['mnist_cvae_model_path'])
ldm.load_state_dict(ldm_dict['model_state_dict'])
ldm.eval()

# ----------------------------
# Set up measurement of spikes
# ----------------------------
indxs = pd.read_pickle(f"{ldm_dict['snn_params']['measurement_indxs_path']}.pkl").loc[(ldm_dict['prob_of_measurement'],ldm_dict['sample_number'])]

# -----------------------------
# Import reference trajectories
# -----------------------------
print('Importing reference trajectory...')
if args.path_to_reference_trajectory.split('/')[-1] == 'set_points':
    ref_traj = np.load(f"{args.path_to_reference_trajectory}/prob_{ldm_dict['prob_of_measurement']}_sample_{ldm_dict['sample_number']}.npy",
                       allow_pickle=True)[()]
elif args.path_to_reference_trajectory.split('/')[-1] == 'arc':
    ref_traj = np.load(f"{args.path_to_reference_trajectory}/ref_traj_{args.arc_num}_prob_{ldm_dict['prob_of_measurement']}_sample_{ldm_dict['sample_number']}.npy",
                       allow_pickle=True)[()]

# -----------------
# Set up controller
# -----------------
print('Creating controller...')
def p_controller(state,ref,p_gains=np.array(args.p_gains)):
    state_error = ref-state
    return p_gains*state_error

# ---------------------
# Initialize collectors
# ---------------------
n_steps = ref_traj.shape[0]
V = np.zeros((n_steps,2))
U = np.zeros((n_steps,28,28))
Z = np.zeros((n_steps,2))
spike_collector = np.zeros((n_steps,len(indxs['raw_indxs'])))

# -----------------
# Set initial guess
# -----------------
Z0 = ref_traj[0,:]

# ------------
# Control Loop
# ------------
for time_step in tqdm(range(n_steps)):
    v_control = p_controller(Z0,ref_traj[time_step,:])
    # Save for inspection
    V[time_step] = v_control.flatten()

    # Project latent input into measurement space (u_n)
    u_control = ldm.u_decoder.decode(torch.from_numpy(v_control).type(torch.float32).reshape(1,-1))
    U[time_step] = u_control.detach().numpy().reshape(28,28)

    # Stimulate SNN with optimal input in measurement space
    sensory_spike_rec, reservoir_spike_rec, output_spike_rec, _ = snn(u_control.reshape(1,-1))

    # Only measure a subset of activity (x_n+1)
    sensory_spikes = sensory_spike_rec.cpu().detach().numpy().reshape(100)[indxs['sensory_indxs']]
    reservoir_spikes = reservoir_spike_rec.cpu().detach().numpy().reshape(500)[indxs['reservoir_indxs']]
    output_spikes = output_spike_rec.cpu().detach().numpy().reshape(10)[indxs['output_indxs']]
    spike_collector[time_step] = np.concatenate((sensory_spikes,reservoir_spikes,output_spikes))

    # Convert to continuous state
    if time_step == 0:
        full_state = torch.from_numpy(spike_collector[time_step]).to(torch.float32)
    else:
        alpha = ldm_dict['snn_params']['ewma_alpha']
        full_state = alpha*torch.from_numpy(spike_collector[time_step]).to(torch.float32)+(1-alpha)*full_state
    # Encode spikes into neural manifold space
    z_np1, _ ,_ = ldm.encode_x(full_state.reshape(1, -1))
    Z[time_step] = z_np1.detach().numpy()

    # Update latent state for MPC optimization
    Z0 = Z[time_step]
    
# Save data
save_dict = {'Z_control':Z,'Z_ref':ref_traj[:],'V':V}
np.save(f"{args.path_to_save_output}/prob_{ldm_dict['prob_of_measurement']}_sample_{ldm_dict['sample_number']}_trial_{args.trial_id}.npy",save_dict)
def nMSE(z_ref,z_control):
    mse = np.mean((z_ref-z_control)**2)
    nmse = mse/(np.max(z_ref)-np.min(z_ref))
    return nmse
z1_nMSE = nMSE(ref_traj[:,0],Z[:,0])
z2_nMSE = nMSE(ref_traj[:,1],Z[:,1])
print(f'Z_1 nMSE: {z1_nMSE}')
print(f'Z_2 nMSE: {z2_nMSE}')
'''

from scipy.ndimage import gaussian_filter1d

def gaussian_smoothing(arr, sigma=10):
    return gaussian_filter1d(arr, sigma)

# Plot state errors
fig,ax = plt.subplots(2,1,sharey=True,sharex=True)
ax[0].plot(ref_traj[:,0],color='black',alpha=0.5)
ax[0].plot(Z[:,0],color='red',alpha=0.5)
ax[0].plot(gaussian_smoothing(Z[:,0]),color='darkred',alpha=0.5)
ax[1].plot(ref_traj[:,1],color='black',alpha=0.5)
ax[1].plot(Z[:,1],color='red',alpha=0.5)
ax[1].plot(gaussian_smoothing(Z[:,1]),color='darkred',alpha=0.5)
plt.show()


plt.plot(V)
plt.show()

# Plot Spikes
plt.imshow(-1*spike_collector.T,aspect='auto',cmap='gray')
plt.show()
breakpoint()


# Plot inputs
for i in range(n_steps):
    plt.cla()
    plt.title(f'Time step {i}/{n_steps}')
    plt.imshow(U[i],aspect='auto')
    plt.pause(.01)
plt.show()

breakpoint()
'''