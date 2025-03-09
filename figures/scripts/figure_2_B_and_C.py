import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from network_architectures import CVAE as model
from stimulus_scripts.load_mnist_data import load_MNIST_data,train_val_test_split
from tqdm import tqdm
from sklearn.cluster import KMeans
# -----------

# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--model_path', default='saved_models/stimulus_vaes/MNIST_VAE', type=str)
p.add_argument('--mnist_data_path',default='mnist_data')
args = p.parse_args()


# -------------------------------------
# Load trained VAE and Model Parameters
# -------------------------------------
model_dict = torch.load(f'{args.model_path}.pt')
params = model_dict['model_params']
model = model.VAE(params)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()
dim_size = params['latent_dim_size']

# --------------------------
# Import V Assimilation Data
# --------------------------
V = np.load('assimilation_data/V_assimilation.npy',allow_pickle=True)[()]
V_train = V['V_train']
V_test = V['V_test']


# Selected values
I = 6000
II = 23960
III = 33100
train_I = V_train[I]
train_II = V_train[II]
train_III = V_train[III]

test_I = V_test[I]
test_II = V_test[II]
test_III = V_test[III]


fig,ax = plt.subplots(2,1,sharex=True,sharey=True,figsize=(10,4))
alpha = 0.7
# Train
ax[0].plot(V_train[:,0],alpha=alpha)
ax[0].plot(V_train[:,1],alpha=alpha)
ax[0].scatter(I,train_I[0],s=10,color='black',alpha=alpha)
ax[0].scatter(I,train_I[1],s=10,color='black',alpha=alpha)
ax[0].scatter(II,train_II[0],s=10,color='black',alpha=alpha)
ax[0].scatter(II,train_II[1],s=10,color='black',alpha=alpha)
ax[0].scatter(III,train_III[0],s=10,color='black',alpha=alpha)
ax[0].scatter(III,train_III[1],s=10,color='black',alpha=alpha)


# Test
ax[1].plot(V_test[:,0],alpha=alpha)
ax[1].plot(V_test[:,1],alpha=alpha)
ax[1].scatter(I,test_I[0],s=10,color='black',alpha=alpha)
ax[1].scatter(I,test_I[1],s=10,color='black',alpha=alpha)
ax[1].scatter(II,test_II[0],s=10,color='black',alpha=alpha)
ax[1].scatter(II,test_II[1],s=10,color='black',alpha=alpha)
ax[1].scatter(III,test_III[0],s=10,color='black',alpha=alpha)
ax[1].scatter(III,test_III[1],s=10,color='black',alpha=alpha)

ax[0].set_yticks([-4,-2,0,2,4])
ax[1].set_yticks([-4,-2,0,2,4])
plt.savefig('figures/raw_figures/Figure_2_B.pdf')

def plot_decoded_images(times):
    fig,ax = plt.subplots(3,2,sharex=True,sharey=True)
    for j,time in enumerate(times):
        train_image = model.decode(torch.from_numpy(V_train[time]).reshape(1, -1).to(torch.float32))
        test_image = model.decode(torch.from_numpy(V_test[time]).reshape(1, -1).to(torch.float32))
        ax[j,0].imshow(train_image.detach().cpu().numpy().reshape((28,28)))
        ax[j,1].imshow(test_image.detach().cpu().numpy().reshape((28,28)))
        ax[j,0].set_yticks([])
        ax[j,0].set_xticks([])
        ax[j,1].set_yticks([])
        ax[j,1].set_xticks([])
    plt.savefig('figures/raw_figures/Figure_2_C.pdf')

plot_decoded_images([I,II,III])
