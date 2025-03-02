import matplotlib.pyplot as plt
import numpy as np
import argparse

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--alpha',default=0.1,type=float)
p.add_argument('--in_data_path',default='assimilation_data/spikes_assimilation')
p.add_argument('--out_data_path',default='assimilation_data/filtered_spikes_assimilation')
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

def lowpass_filter(data,alpha):
    time_steps = data.shape[1]
    cs_data = np.zeros_like(data)
    for i in range(time_steps):
        if i == 0:
            cs_data[:,i] = data[:,i]
        else:
            cs_data[:,i] = cs_data[:,i-1]+alpha*(data[:,i]-cs_data[:,i-1])
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
    ewma_output_data = {'sensory_spikes': lowpass_filter(sensory_spikes,args.alpha),
                        'reservoir_spikes': lowpass_filter(reservoir_spikes,args.alpha),
                        'output_spikes': lowpass_filter(output_spikes,args.alpha)}
    # Save to out_data dict
    out_data[f'{split}'] = ewma_output_data

# Save EWMA filtering hyperparameter
out_data['alpha'] = args.alpha

# ---------
# Save Data
# ---------
np.save(f'{args.out_data_path}.npy',out_data)
num_plots = 5
fig,ax = plt.subplots(num_plots,3,sharex=True,sharey=True)
for i in range(num_plots):
    ax[i,0].plot(out_data['X_train']['reservoir_spikes'][i])
    ax[i,1].plot(out_data['X_val']['reservoir_spikes'][i])
    ax[i,2].plot(out_data['X_test']['reservoir_spikes'][i])
plt.show()

