import numpy as np
import pandas as pd
import argparse

# Parse Args
p = argparse.ArgumentParser()
p.add_argument('--path_to_p_data', type=str, default='./neural_manifold_control/reactive_control/pid_control/set_point_control')
p.add_argument('--path_to_mpc_data', type=str, default='./neural_manifold_control/mpc/set_point_control')
args = p.parse_args()

def nMSE(z_ref, z_control):
    mse = np.mean((z_ref - z_control) ** 2)
    nmse = mse / (np.max(z_ref) - np.min(z_ref))
    return nmse

def RMS(V):
    return np.sqrt(np.mean(V ** 2))

probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
n_samples = 10
n_trials = 50

data_list = []

for prob in probs:
    for sample in range(n_samples):
        for trial in range(n_trials):
            p_file = f'{args.path_to_p_data}/prob_{prob}_sample_{sample}_trial_{trial}.npy'
            mpc_file = f'{args.path_to_mpc_data}/prob_{prob}_sample_{sample}_trial_{trial}.npy'
            
            p_data = np.load(p_file, allow_pickle=True)[()]
            mpc_data = np.load(mpc_file, allow_pickle=True)[()]

            # Get reference trajectory
            Z_ref = mpc_data['Z_ref']

            # Store results for PID controller
            data_list.append(['pid', prob, sample, trial, nMSE(Z_ref[:, 0], p_data['Z_control'][:, 0])])
            data_list.append(['pid', prob, sample, trial, nMSE(Z_ref[:, 1], p_data['Z_control'][:, 1])])
            
            # Store results for MPC controller
            data_list.append(['mpc', prob, sample, trial, nMSE(Z_ref[:, 0], mpc_data['Z_control'][:, 0])])
            data_list.append(['mpc', prob, sample, trial, nMSE(Z_ref[:, 1], mpc_data['Z_control'][:, 1])])

# Convert to DataFrame
df = pd.DataFrame(data_list, columns=['controller', 'observation probability', 'sample', 'trial', 'nMSE'])

# Display the first few rows
print(df.head())
breakpoint()