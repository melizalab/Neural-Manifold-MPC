import numpy as np
import matplotlib.pyplot as plt
import argparse

plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 7,     # X-axis tick labels
    'ytick.labelsize': 7,     # Y-axis tick labels
})
# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_mpc_data',type=str,default='./neural_manifold_control/mpc')
args = p.parse_args()

# Functions
def nMSE(z_ref,z_control):
    mse = np.mean((z_ref-z_control)**2)
    nmse = mse/(np.max(z_ref)-np.min(z_ref))
    return nmse

def RMS(V):
    return np.sqrt(np.mean(V**2))

def get_arcs(ref_path,trial):
    arc_data = np.load(f'{args.path_to_mpc_data}/{ref_path}_control/prob_0.2_sample_0_trial_{trial}.npy',allow_pickle=True)[()]
    Z_control = arc_data['Z_control']
    Z_ref = arc_data['Z_ref']
    V = arc_data['V']
    return Z_control,Z_ref,V

# Plot params
n_trials = 50
arc_1_color='darkcyan'
arc_2_color='darkmagenta'
v1_color = '#1f77b4'
v2_color = '#ff7f0e'
alpha = 0.05
linewidth=.1


# nMSEs
z1_nMSE_arc_1 = []
z2_nMSE_arc_1 = []
z1_nMSE_arc_2 = []
z2_nMSE_arc_2 = []

v1_RMS_arc_1 = []
v2_RMS_arc_1 = []
v1_RMS_arc_2 = []
v2_RMS_arc_2 = []


for trial in range(n_trials):
    Z_control_arc_1,Z_ref_arc_1,V_arc_1 = get_arcs(ref_path='arc_1',trial=trial)
    Z_control_arc_2,Z_ref_arc_2,V_arc_2 = get_arcs(ref_path='arc_2',trial=trial)

    # Get nMSEs
    z1_nMSE_arc_1.append(nMSE(Z_control_arc_1[:,0],Z_ref_arc_1[:,0]))
    z2_nMSE_arc_1.append(nMSE(Z_control_arc_1[:,1],Z_ref_arc_1[:,1]))

    z1_nMSE_arc_2.append(nMSE(Z_control_arc_2[:,0],Z_ref_arc_2[:,0]))
    z2_nMSE_arc_2.append(nMSE(Z_control_arc_2[:,1],Z_ref_arc_2[:,1]))

    # Get RMS
    v1_RMS_arc_1.append(RMS(V_arc_1[:,0]))
    v2_RMS_arc_1.append(RMS(V_arc_1[:,1]))
    v1_RMS_arc_2.append(RMS(V_arc_2[:,0]))
    v2_RMS_arc_2.append(RMS(V_arc_2[:,1]))




fig, ax = plt.subplots(2, 1, sharex=True, figsize=(1, 3))

# Data for boxplots
z_data = [z1_nMSE_arc_1, z2_nMSE_arc_1, z1_nMSE_arc_2, z2_nMSE_arc_2]
v_data = [v1_RMS_arc_1, v2_RMS_arc_1, v1_RMS_arc_2, v2_RMS_arc_2]
pos = [-0.25, 0.25, 1.25, 1.75]  # Positions

# Define colors and patterns
z_colors = [arc_1_color, arc_1_color, arc_2_color, arc_2_color]  # Matching arc colors
z_hatches = ['//', 'xxx', '//', 'xxx']  # Distinguish z1 and z2

v_colors = [v1_color, v2_color, v1_color, v2_color]  # Matching velocity colors

# Create boxplots
z_bp = ax[0].boxplot(z_data, positions=pos, patch_artist=True,
                      medianprops={'color': 'black', 'linewidth': 1.5},
                      whiskerprops={'linewidth': 1.2},
                      capprops={'linewidth': 1.2},
                      flierprops={'marker': 'o', 'markersize': 2, 'markeredgewidth': 1})

v_bp = ax[1].boxplot(v_data, positions=pos, patch_artist=True,
                      medianprops={'color': 'black', 'linewidth': 1.5},
                      whiskerprops={'linewidth': 1.2},
                      capprops={'linewidth': 1.2},
                      flierprops={'marker': 'o', 'markersize': 2, 'markeredgewidth': 1})

# Apply colors and hatches to z boxes
for box, color, hatch in zip(z_bp['boxes'], z_colors, z_hatches):
    box.set(facecolor=color, alpha=1.0, hatch=hatch, edgecolor='black')

# Apply colors to v boxes, whiskers, and medians
for i, (box, color) in enumerate(zip(v_bp['boxes'], v_colors)):
    box.set(facecolor=color, alpha=0.5, edgecolor='black')  # Box color

    # Set the whisker and cap color for each box
    whiskers = v_bp['whiskers'][2*i:2*i+2]  # Select whiskers for each box
    caps = v_bp['caps'][2*i:2*i+2]  # Select caps for each box
    for whisker in whiskers:
        whisker.set(color=color, linewidth=1.5)  # Set whisker color
    for cap in caps:
        cap.set(color=color, linewidth=1.5)  # Set cap color

ax[0].set_xticks([0,1.5])
ax[0].set_ylabel('nMSE')
ax[1].set_xticklabels(['Traj 1','Traj 2'])
ax[1].set_ylabel('RMS')
plt.savefig('figures/raw_figures/figure_5_nMSE_RMS.pdf')
