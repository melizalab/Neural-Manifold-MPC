import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import argparse


# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--path_to_forecasts',type=str,default='assimilation_data/Z_assimilation')
p.add_argument('--prob_of_measurement',default=.2,type=float)
p.add_argument('--sample_number',default=0,type=int)
p.add_argument('--num_time_steps',type=int,default=1000)
p.add_argument('--save_path',type=str,default='reference_trajectories/set_points')
args = p.parse_args()

# ------------------
# Load Forecast Data
# ------------------
forecast_data = np.load(f'{args.path_to_forecasts}/prob_{args.prob_of_measurement}_sample_{args.sample_number}.npy',allow_pickle=True)[()]
Z_test = forecast_data['Z_test']
# ------------------------
# K-Means on Training Data
# ------------------------
print('K-MEANS on Latents...')
n_centers = 2
kmeans = KMeans(n_clusters=n_centers, random_state=0,n_init=10).fit(Z_test)
centers = kmeans.cluster_centers_

# ---------------------------
# Create reference trajectory
# ---------------------------
ref_traj = np.repeat(centers,args.num_time_steps//2,axis=0)
fig,ax = plt.subplots(1,2)
ax[0].scatter(Z_test[:,0],Z_test[:,1],s=.2,alpha=.5)
ax[0].scatter(centers[0,0],centers[0,1],color='black')
ax[0].scatter(centers[1,0],centers[1,1],color='black')
ax[1].plot(ref_traj)
plt.show()
# -------------------------
# Save reference trajectory
# -------------------------
np.save(f'{args.save_path}/prob_{args.prob_of_measurement}_sample_{args.sample_number}.npy',ref_traj)
