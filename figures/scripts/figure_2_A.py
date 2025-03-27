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


plt.rcParams.update({
    'axes.labelsize': 8,      # Axis labels
    'xtick.labelsize': 6,     # X-axis tick labels
    'ytick.labelsize': 6,     # Y-axis tick labels
})
# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device('cpu')
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
print('LOADING DATA...')
mnist_data,_ = load_MNIST_data(data_path=args.mnist_data_path,download=False)
_,_,mnist_test = train_val_test_split(mnist_data,*params['train_val_test_ratios'],seed=params['training_random_seed'])
test_loader = DataLoader(dataset=mnist_test,batch_size=len(mnist_test),shuffle=True,drop_last=True)

# --------------------------------------
# Get latent representation of test data
# --------------------------------------
for (data,labels) in tqdm(test_loader):
    with torch.no_grad():
        _, latent_rep, latent_logvar = model(data.to(device))

# ----------------------------------------
# Get centers used for V assimilation data
# ----------------------------------------
kmeans = KMeans(n_clusters=120, random_state=0,n_init=10).fit(latent_rep.detach().cpu().numpy())
centers = kmeans.cluster_centers_

# ----
# Plot
# ----
plt.figure(figsize=(5, 5))

plt.scatter(latent_rep[:, 0], latent_rep[:, 1], c=labels, cmap='tab10', s=.5, alpha=0.7) # Test embeddings
plt.scatter(centers[80:,0],centers[80:,1],color='black',s=10,marker='d') # Centers

plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.title("Latent Space Representation of MNIST Digits")
plt.xticks([-4,-2,0,2,4])
plt.yticks([-4,-2,0,2,4])
plt.xlim([-4.5,4.5])
plt.ylim([-4.5,4.5])
plt.xlabel('V1')
plt.ylabel('V2')
plt.savefig('figures/raw_figures/Figure_2_A.pdf')
breakpoint()

