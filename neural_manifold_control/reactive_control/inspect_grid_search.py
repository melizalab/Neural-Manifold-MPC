import numpy as np
import matplotlib.pyplot as plt
import glob


def nMSE(z_ref,z_control):
    mse = np.mean((z_ref-z_control)**2)
    nmse = mse/(np.max(z_ref)-np.min(z_ref))
    return nmse

file_path = 'neural_manifold_control/reactive_control/p_control/grid_search/*'
best_m_nmse = float('inf')
best_file = None

for file in glob.glob(file_path):
    data = np.load(file,allow_pickle=True)[()]
    Z_ref = data['Z_ref']
    Z_control = data['Z_control']
    Z1_nmse = nMSE(Z_ref[:,0],Z_control[:,0])
    Z2_nmse = nMSE(Z_ref[:,1],Z_control[:,1])
    m_nmse = (Z1_nmse+Z2_nmse)/2
    if m_nmse < best_m_nmse:
        best_z1 = Z1_nmse
        best_z2 = Z2_nmse
        best_m_nmse = m_nmse
        best_file = file
best_data = np.load(best_file,allow_pickle=True)[()]
print(best_z1,best_z2)
print(best_file)
ref_traj = best_data['Z_ref']
Z = best_data['Z_control']
fig,ax = plt.subplots(2,1,sharey=True,sharex=True)
ax[0].plot(ref_traj[:,0],color='black',alpha=0.5)
ax[0].plot(Z[:,0],color='red',alpha=0.5)
ax[1].plot(ref_traj[:,1],color='black',alpha=0.5)
ax[1].plot(Z[:,1],color='red',alpha=0.5)
plt.show()