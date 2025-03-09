import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 7,     # X-axis tick labels
    'ytick.labelsize': 7,     # Y-axis tick labels
})

file_path = 'figures/data/spikes_and_latents_for_prob_0.2_sample_0'
data = np.load(f'{file_path}.npy',allow_pickle=True)[()]
z1_corr = pearsonr(data['Z_test'][:,0],data['Z_hat_test'][:,0])[0]
z2_corr = pearsonr(data['Z_test'][:,1],data['Z_hat_test'][:,1])[0]
x_corr = pearsonr(data['X_test'][:,0],data['X_hat_test'][:,0])[0]
breakpoint()

fig,ax = plt.subplots(2,2,figsize=(4,2))
linewidth=1
# Z1 Test
ax[0,0].plot(data['Z_test'][:,0],color='black',alpha=0.7,linewidth=linewidth)
ax[0,0].plot(data['Z_hat_test'][:,0],color='darkred',alpha=0.7,linewidth=linewidth)
ax[0,0].set_ylim([-.12,.12])
ax[0,0].set_xticks([0,10000,20000,30000])

# Z2 Test
ax[1,0].plot(data['Z_test'][:,1],color='black',alpha=0.7,linewidth=linewidth)
ax[1,0].plot(data['Z_hat_test'][:,1],color='darkred',alpha=0.7,linewidth=linewidth)
ax[1,0].set_ylim([-.12,.12])
ax[1,0].set_xticks([0,10000,20000,30000])

# X Test
ax[0,1].imshow(data['X_test'].T,aspect='auto')
ax[0,1].set_xlim([0,3000])
ax[1,1].imshow(data['X_hat_test'].T,aspect='auto')
ax[1,1].set_xlim([0,3000])
ax[1,1].set_yticks([])
plt.savefig('figures/raw_figures/figure_3_forecasts.pdf')


fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(1,2))
ax[0].scatter(data['Z_train'][::10,0],data['Z_train'][::10,1],s=.01,alpha=0.2)
ax[1].scatter(data['Z_test'][::10,0],data['Z_test'][::10,1],s=.01,alpha=0.2)
ax[1].set_xlim([-.12,.12])
ax[1].set_ylim([-.12,.12])
plt.savefig('figures/raw_figures/figure_3_embeddings.pdf')
