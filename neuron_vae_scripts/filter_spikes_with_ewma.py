import matplotlib.pyplot as plt
import numpy as np
import argparse

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--alpha',default=0.5,type=float)
p.add_argument('--in_data_path',default='latent_control/assimilation_data/binary_assimilation_spikes')
p.add_argument('--out_data_path',default='latent_control/assimilation_data/continuous_assimilation_spikes')
args = p.parse_args()

# --------------------------------------------
# Exponentially Weighted Moving Average Filter
# --------------------------------------------
def EWMA(data,alpha):
    time_steps = data.shape[1]
    cs_data = np.zeros_like(data)
    for i in range(time_steps):
        if i == 0:
            cs_data[:,i] = data[:,i]
        else:
            cs_data[:,i] = alpha*data[:,i]+(1-alpha)*cs_data[:,i-1]
    return cs_data

# -------------------------------
# Compute for Train and Test Data
# -------------------------------
in_data = np.load(f'{args.in_data_path}.npy',allow_pickle=True)[()]
out_data = {}
for split in in_data.keys():
    # Load data
    sensory_spikes = in_data[split]['sensory_spikes']
    reservoir_spikes = in_data[split]['reservoir_spikes']
    output_spikes = in_data[split]['output_spikes']
    # Apply EWMA Filter to Spikes
    ewma_output_data = {'sensory_spikes': EWMA(sensory_spikes,args.alpha),
                        'reservoir_spikes': EWMA(reservoir_spikes,args.alpha),
                        'output_spikes': EWMA(output_spikes,args.alpha)}
    # Save to out_data dict
    out_data[f'{split}'] = ewma_output_data

# ---------
# Save Data
# ---------
num_plots = 10
fig,ax = plt.subplots(num_plots,1,sharex=True,sharey=True)
for i in range(num_plots):
    ax[i].plot(out_data['X_train']['reservoir_spikes'][i])
plt.show()
np.save(f'{args.out_data_path}.npy',out_data)
