import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import argparse

# ------------------------------
# Append path for custom modules
# ------------------------------
from .load_mnist_data import load_MNIST_data,train_val_test_split
from network_architectures import CVAE as model
from .interpolate_latent_stimuli import get_latent_trajectory

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--model_path', default='saved_models/stimulus_vaes/MNIST_VAE', type=str)
p.add_argument('--mnist_data_path',default='mnist_data',type=str)
p.add_argument('--n_centers',default=120,type=int) # 33% train, 33% val, 33% test
p.add_argument('--step_len',default=500,type=int)
p.add_argument('--slow_len',default=1000,type=int)
p.add_argument('--fast_len',default=200,type=int)
p.add_argument('--out_path',default='assimilation_data/V_assimilation',type=str)
p.add_argument('--save', action='store_true', help="save V_assimilation")
args = p.parse_args()

# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')

# -------------------------------------
# Load trained VAE and Model Parameters
# -------------------------------------
model_dict = torch.load(f'{args.model_path}.pt')
params = model_dict['model_params']
model = model.VAE(params).to(device)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()
dim_size = params['latent_dim_size']

# -----------------------
# Load MNIST testing data
# -----------------------
print('Loading MNIST data...')
mnist_data,_ = load_MNIST_data(data_path=args.mnist_data_path,download=False)
_,_,mnist_test = train_val_test_split(mnist_data,*params['train_val_test_ratios'],seed=params['training_random_seed'])
test_loader = DataLoader(dataset=mnist_test,batch_size=len(mnist_test),shuffle=True,drop_last=True)

# ---------------------------------------------
# Get latent representation (V) of MNIST digits
# ---------------------------------------------
print('Encoding Images...')
MU = torch.zeros((params['batch_size'],params['latent_dim_size']))
LOGVAR = torch.zeros_like(MU)
for _,(data,labels) in enumerate(test_loader):
    with torch.no_grad():
        _, latent_rep, _ = model(data.to(device))

# ------------
# K-Means on V
# ------------
print('K-MEANS on Latents...')
kmeans = KMeans(n_clusters=args.n_centers, random_state=0,n_init=10).fit(latent_rep.detach().cpu().numpy())
centers = kmeans.cluster_centers_

# -----------------
# Plot Latent Space
# -----------------
print('Plotting Latent Space and Centers...')
plt.scatter(latent_rep[:,0].detach().cpu(),latent_rep[:,1].detach().cpu(),c=labels.numpy(),cmap='viridis',s=0.5)
plt.scatter(centers[:,0],centers[:,1],color='black',s=10,marker='D')
plt.show()

# ------------------------
# Decode Sample of Centers
# ------------------------
print('Plotting Ten Decoded Centers...')
n_plot = 10
fig,ax = plt.subplots(2,n_plot//2)
for i in range(n_plot):
    image = model.decode(torch.from_numpy(centers[i]).reshape(1,-1).to(device))
    if i <5:
        row = 0
    else:
        row = 1
    ax[row,i%5].imshow(image.detach().cpu().numpy().reshape(28,28))
    ax[row,i%5].set_xticks([])
    ax[row,i%5].set_yticks([])
plt.show()

# ----------------------
# Make Latent Trajectory
# ----------------------
# TODO: Make this work for arbitrary number of centers
train_centers = centers[:40,:]
val_centers = centers[40:80,:]
test_centers = centers[80:,:]

print('Calculating Latent Trajectories...')
V_train = get_latent_trajectory(train_centers,
                                train_val_test='train',
                                step_len=args.step_len,
                                slow_len=args.slow_len,
                                fast_len=args.fast_len,
                                params=params)
V_val = get_latent_trajectory(val_centers,
                              train_val_test='val',
                              step_len=args.step_len,
                              slow_len=args.slow_len,
                              fast_len=args.fast_len,
                              params=params)
V_test = get_latent_trajectory(test_centers,
                               train_val_test='test',
                               step_len=args.step_len,
                               slow_len=args.slow_len,
                               fast_len=args.fast_len,
                               params=params)

print('Plotting V_assimilation data...')
fig,ax=plt.subplots(2,3,sharex=True,sharey=True)
for i in range(2):
    ax[i,0].plot(V_train[:,i])
    ax[i,1].plot(V_val[:,i])
    ax[i,2].plot(V_test[:,i])
ax[0,0].set_title('Train')
ax[0,1].set_title('Val')
ax[0,2].set_title('Test')
ax[0,0].set_ylabel('V1')
ax[1,0].set_ylabel('V2')
plt.show()
if args.save == True:
    print('Saving...')
    V_assimilation = {'V_train': V_train, 'V_val':V_val, 'V_test': V_test}
    np.save(f'{args.out_path}.npy',V_assimilation)


