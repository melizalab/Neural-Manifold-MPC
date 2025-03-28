import numpy as np
import matplotlib.pyplot as plt
import glob
import argparse


# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
#p.add_argument('--path_to_data',type=str,default='neural_manifold_control/reactive_control/pid_control/set_point_control')
p.add_argument('--path_to_data',type=str,default='neural_manifold_control/mpc/set_point_control')

p.add_argument('--prob_of_measurement',default=.3,type=float)
p.add_argument('--sample_number',default=0,type=int)
p.add_argument('--num_time_steps',type=int,default=1000)
args = p.parse_args()

def nMSE(z_ref,z_control):
    mse = np.mean((z_ref-z_control)**2)
    nmse = mse/(np.max(z_ref)-np.min(z_ref))
    return nmse

fig,ax = plt.subplots(2,2)
alpha = 0.03
Z_mean = np.zeros((1000,2))

z1_nMSE_mean = 0
z2_nMSE_mean = 0

# Loop through files
num_files = 0
for file in glob.glob(f'{args.path_to_data}/prob_{args.prob_of_measurement}_sample_{args.sample_number}_trial*'):
    num_files+=1
    data = np.load(file,allow_pickle=True)[()]
    Z_control = data['Z_control']
    Z_ref = data['Z_ref']

    Z_mean = Z_mean+Z_control

    V = data['V']
    z1_nMSE = nMSE(Z_ref[:,0],Z_control[:,0])
    z2_nMSE = nMSE(Z_ref[:,1],Z_control[:,1])
    z1_nMSE_mean+= z1_nMSE
    z2_nMSE_mean += z2_nMSE

    # Plot
    ax[0,0].plot(Z_control[:,0],color='red',alpha=alpha)
    ax[1,0].plot(Z_control[:,1],color='red',alpha=alpha)
    ax[0,0].set_title('Z Latent State')

    ax[0,1].plot(V[:,0],color='blue',alpha=alpha)
    ax[1,1].plot(V[:,1],color='blue',alpha=alpha)
    ax[0,1].set_title('V Latent Input')
    if num_files == 10:
        break
z1_nMSE_mean/=num_files 
z2_nMSE_mean/=num_files
print(f'z1 nMSE mean: {z1_nMSE_mean:4f}, z2 nMSE mean: {z2_nMSE_mean:4f}') 

Z_mean = Z_mean/num_files
ax[0,0].plot(Z_ref[:,0],color='black',alpha=0.5,label='ref traj')
ax[1,0].plot(Z_ref[:,1],color='black',alpha=0.5, label = 'ref traj')
ax[0,0].plot(Z_mean[:,0],color='darkred',alpha=0.5)
ax[1,0].plot(Z_mean[:,1],color='darkred',alpha=0.5)
ax[0,0].set_ylabel('Z1')
ax[1,0].set_ylabel('Z2')
ax[0,1].set_ylabel('V1')
ax[1,1].set_ylabel('V2')
plt.tight_layout()
print(f'Num files: {num_files}')
plt.show()