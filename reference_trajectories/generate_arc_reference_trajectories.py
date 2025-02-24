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
p.add_argument('--save_path',type=str,default='reference_trajectories/arcs')
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
x1,y1 = centers[0]
x2,y2 = centers[1]
midpoint = np.array([0.5*(x1+x2),0.5*(y1+y2)])
h,k = midpoint
r = np.sqrt((x1-h)**2+(y1-k)**2)
theta1 = np.arctan2(y1,x1)
theta2 = theta1+np.pi
t = np.linspace(0,1,args.num_time_steps)
a0 = theta1
a1 = theta2
a = a1-a0

arc1_x = h+r*np.cos(a0+a*t)
arc1_y = k+r*np.sin(a0+a*t)
ref_traj_1 = np.array([arc1_x,arc1_y]).T

arc2_x = h+r*np.cos(a0+a*t-np.pi)
arc2_y = k+r*np.sin(a0+a*t-np.pi)
ref_traj_2 = np.array([arc2_x[::-1],arc2_y[::-1]]).T

plt.scatter(Z_test[:,0],Z_test[:,1],s=.2,alpha=.5)
plt.scatter(centers[0,0],centers[0,1],color='black')
plt.scatter(centers[1,0],centers[1,1],color='black')
plt.plot(ref_traj_1[:,0],ref_traj_1[:,1],label='ref traj 1')
plt.plot(ref_traj_2[:,0],ref_traj_2[:,1],label='ref traj 2')
plt.legend()
plt.show()

# -------------------------
# Save reference trajectory
# -------------------------
np.save(f'{args.save_path}/ref_traj_1_prob_{args.prob_of_measurement}_sample_{args.sample_number}.npy',ref_traj_1)
np.save(f'{args.save_path}/ref_traj_2_prob_{args.prob_of_measurement}_sample_{args.sample_number}.npy',ref_traj_2)
