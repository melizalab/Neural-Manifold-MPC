import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import gaussian_kde


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

def nMSE(z_ref,z_control):
    mse = np.mean((z_ref-z_control)**2)
    nmse = mse/(np.max(z_ref)-np.min(z_ref))
    return nmse

n_trials = 50
arc_1_color='darkcyan'
arc_2_color='darkmagenta'
unconst_color='darkred'
alpha = 0.1
linewidth=.1
sample = 0
plt.figure(figsize=(2.4,2.4))

arc_1_mean = np.zeros((1000,2))
arc_2_mean = np.zeros((1000,2))
unconst_mean = np.zeros((1000,2))

def get_arcs(ref_path,trial):
    arc_data = np.load(f'{args.path_to_mpc_data}/{ref_path}_control/prob_0.2_sample_{sample}_trial_{trial}.npy',allow_pickle=True)[()]
    Z_control = arc_data['Z_control']
    Z_ref = arc_data['Z_ref']
    return Z_control,Z_ref
# Get latent embeddings
forecast_data = np.load(f'assimilation_data/Z_assimilation/prob_0.2_sample_{sample}.npy',allow_pickle=True)[()]
Z_test = forecast_data['Z_test']

for trial in range(n_trials):
    Z_control_arc_1,Z_ref_arc_1 = get_arcs(ref_path='arc_1',trial=trial)
    Z_control_arc_2,Z_ref_arc_2 = get_arcs(ref_path='arc_2',trial=trial)
    Z_control_unconst,Z_ref_unconst = get_arcs(ref_path='set_point',trial=trial)

    arc_1_mean += Z_control_arc_1
    arc_2_mean += Z_control_arc_2
    unconst_mean += Z_control_unconst
    
    #plt.plot(Z_control_arc_1[:,0],Z_control_arc_1[:,1],color=arc_1_color,alpha=alpha,linewidth=linewidth)
    #plt.plot(Z_control_arc_2[:,0],Z_control_arc_2[:,1],color=arc_2_color,alpha=alpha,linewidth=linewidth)
    #plt.plot(Z_control_unconst[:,0],Z_control_unconst[:,1],color=unconst_color,alpha=alpha,linewidth=linewidth)

plt.plot(Z_ref_arc_1[:,0],Z_ref_arc_1[:,1],color='k')
plt.plot(Z_ref_arc_2[:,0],Z_ref_arc_2[:,1],color='k',linestyle='--')
arc_1_mean/=n_trials
arc_2_mean/=n_trials
unconst_mean/=n_trials
mean_linewidth = .5
offset = 25
downsample = 1
plt.plot(arc_1_mean[offset:,0],arc_1_mean[offset:,1],color=arc_1_color,linewidth=mean_linewidth)
plt.plot(arc_2_mean[offset:,0],arc_2_mean[offset:,1],color=arc_2_color,linewidth=mean_linewidth)
plt.plot(unconst_mean[offset:,0],unconst_mean[offset:,1],color=unconst_color,linewidth=mean_linewidth)

plt.scatter(Z_ref_arc_1[0,0],Z_ref_arc_1[0,1],color='black',marker='x',s=15,zorder=10)
plt.scatter(Z_ref_arc_1[-1,0],Z_ref_arc_1[-1,1],color='black',marker='x',s=15,zorder=10)

# Get contour plot of latent embeddings
# Extract x and y coordinates
z1,z2 = Z_test[:, 0], Z_test[:, 1]

# Compute KDE
kde = gaussian_kde(np.vstack([z1, z2]))
z1_i, z2_i = np.meshgrid(
    np.linspace(-0.08, 0.08, 50),
    np.linspace(-0.08, 0.08, 50)
)
zi = kde(np.vstack([z1_i.ravel(), z2_i.ravel()])).reshape(z1_i.shape)

# Create contour plot
plt.contourf(z1_i,z2_i, zi, levels=10,cmap='coolwarm', alpha=.5)
plt.xticks([-0.06,-0.03,0,0.03,0.06])
plt.yticks([-0.06,-0.03,0,0.03,0.06])

plt.xlim([-0.08,0.08])
plt.ylim([-0.08,0.08])
plt.savefig('figures/raw_figures/figure_6_A_state_space.pdf')

