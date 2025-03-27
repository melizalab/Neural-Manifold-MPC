import numpy as np
import matplotlib.pyplot as plt
from network_architectures.latent_linear_dynamics import LDM

file_path = 'figures/data'
probs = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
n_samples = 10
plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 6,     # X-axis tick labels
    'ytick.labelsize': 6,     # Y-axis tick labels
})
# Store mean and standard deviation
z1_test_means, z1_test_stds = [], []
z2_test_means, z2_test_stds = [], []
x_test_means, x_test_stds = [], []
x_decoded_test_means, x_decoded_test_stds = [], []

x_color = 'darkgoldenrod'
z1_color = 'darkcyan'
z2_color = 'darkmagenta'
alpha = 0.5
linewidth = 1

fig, ax = plt.subplots(1, figsize=(2, 2))

for i, prob in enumerate(probs):
    z1_test_corrs = []
    z2_test_corrs = []
    x_test_corrs = []
    x_test_decoded_corrs = []

    for sample in range(n_samples):
        data = np.load(f'figures/data/prob_{prob}_sample_{sample}.npy', allow_pickle=True)[()]

        z1_test_corrs.append(data['z1_test_corr'])
        z2_test_corrs.append(data['z2_test_corr'])
        x_test_corrs.append(data['X_test_corr'])
        x_test_decoded_corrs.append(data['X_test_decoded_corr'])

    # Compute mean and std deviation
    z1_test_means.append(np.mean(z1_test_corrs))
    z1_test_stds.append(np.std(z1_test_corrs))
    
    z2_test_means.append(np.mean(z2_test_corrs))
    z2_test_stds.append(np.std(z2_test_corrs))
    
    x_test_means.append(np.mean(x_test_corrs))
    x_test_stds.append(np.std(x_test_corrs))

    x_decoded_test_means.append(np.mean(x_test_decoded_corrs))
    x_decoded_test_stds.append(np.std(x_test_decoded_corrs))

# Plot with error bars
ax.errorbar(np.arange(8), z1_test_means, yerr=z1_test_stds, color=z1_color, linestyle='-', marker='o', alpha=alpha, capsize=2,markersize=2,linewidth=linewidth)
ax.errorbar(np.arange(8), z2_test_means, yerr=z2_test_stds, color=z2_color, linestyle='-', marker='o', alpha=alpha, capsize=2,markersize=2,linewidth=linewidth)
ax.errorbar(np.arange(8), x_test_means, yerr=x_test_stds, color=x_color, linestyle='-', marker='o', alpha=alpha, capsize=2,markersize=2,linewidth=linewidth)
ax.errorbar(np.arange(8), x_decoded_test_means, yerr=x_decoded_test_stds, color=x_color, linestyle='--', marker='o', alpha=alpha, capsize=2,markersize=2,linewidth=linewidth)

ax.set_ylim([-0.1, 1.1])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(range(len(probs)))
ax.set_xticklabels([.01, .05, .1, .2, .3, .4, .5, .6])
ax.set_xlabel("Probability")
ax.set_ylabel("Correlation")
plt.tight_layout()

plt.savefig('figures/raw_figures/figure_5_A_corr_across_prob.pdf')
