import sys
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
from scipy.stats import pearsonr

# ------------------------------
# Append path for custom modules
# ------------------------------
from network_architectures import VAE as nets
from .load_SNN_data_for_VAE import VAE_dataloader,concatenate_spikes

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument("--X_assimilation_path", default='latent_control/assimilation_data/continuous_assimilation_spikes',type=str)
p.add_argument("--measurement_indxs_path", default='latent_control/assimilation_data/measurement_indxs',type=str)
p.add_argument('--prob_of_measurement',default=.2,type=float)
p.add_argument('--sample_number',default=0,type=int)
p.add_argument('--hidden_layer_sizes',nargs='+',default=[1000,1000], type = int)
p.add_argument("--latent_dim_size", default=2, type=int)
p.add_argument("--batch_size", default=256, type=int)
p.add_argument("--num_epochs", default=20, type=int)
p.add_argument("--print_accuracy_iter", default=90, type=int)
p.add_argument('--kl_scaling',default=0.001,type=float)
p.add_argument('--learning_rate',default=1e-3,type=float)
p.add_argument('--decay_rate',default=0.99,type=float)
p.add_argument('--eps_scaling',default=1.0,type=float)
p.add_argument('--num_examples',default=4000,type=int)
# Path to Save VAE
p.add_argument("--save_path", default='latent_control/saved_models/snn_vaes/SNN_VAE')
args = p.parse_args()

# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')

# -------------
# Load SNN Data
# -------------
print('Loading data...')
full_data = np.load(f'{args.X_assimilation_path}.npy',allow_pickle=True)[()]
measured_spikes = concatenate_spikes(spike_data=full_data['X_train'],
                                     measurement_indxs_path=args.measurement_indxs_path,
                                     prob=args.prob_of_measurement,
                                     sample=args.sample_number)


#train_loader, input_size, VAE_data = load_VAE_data(full_data['X_train'],f'{args.measurement_indxs_path}.npy',batch_size=args.batch_size)
train_loader,input_size = VAE_dataloader(measured_spikes=measured_spikes,
                                         batch_size=args.batch_size,
                                         shuffle=True)
print(f'Number of neurons recorded from: {input_size}')

# -------------------
# Get VAE parameters
# ------------------
params = {'input_size':input_size,
          'latent_dim_size':args.latent_dim_size,
          'hidden_layer_sizes':args.hidden_layer_sizes,
          'eps_scaling':args.eps_scaling,
          'batch_size':args.batch_size,
          'kl_scaling':args.kl_scaling,
          'num_epochs':args.num_epochs,
          'learning_rate':args.learning_rate,
          'deacy_rate':args.decay_rate}

# ------------
# Instance VAE
# ------------
net = nets.VAE(params).to(device)
net.train()

# -----------------
# VAE loss functions
# -----------------
def continuous_loss_function(output,x,mu,logvar,kl_scaling):
    MSE = nn.MSELoss(reduction='sum')
    recon_loss = MSE(x,output)
    kl_loss = -kl_scaling*torch.sum(1+logvar-mu**2-logvar.exp())
    return recon_loss,kl_loss


def binary_loss_function(output,x,mu,logvar,kl_scaling):
    recon_loss = F.binary_cross_entropy(output, x, reduction='sum') / x.size(0)
    kl_loss = -kl_scaling*torch.sum(1+logvar-mu**2-logvar.exp())/x.size(0)
    return recon_loss,kl_loss

# ----------------
# Set up optimizer
# ----------------
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
lmbda = lambda epoch: args.decay_rate**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

# ---------
# Train VAE
# ---------
print('Training VAE...')
train_loss = []
for epoch in range(args.num_epochs):
    counter = 0
    for i,X in enumerate(train_loader):

        # Get batch and predicted values
        X_hat,mu,logvar = net(X.to(device))

        # Calculate batch loss
        recon_loss, kl_loss = binary_loss_function(X_hat, X.reshape(args.batch_size, -1).to(device), mu, logvar,kl_scaling=args.kl_scaling)
        batch_loss = recon_loss+kl_loss

        # Backpropagate
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Append losses
        train_loss.append(batch_loss.detach().cpu().numpy())

        # Print accuracy
        counter += 1
        if counter % args.print_accuracy_iter == 0:
            print(f'Epoch: {epoch+1}/{args.num_epochs}, '
                  f'iteration: {counter}/{len(train_loader)}, '
                  f'train loss: {batch_loss:.2f}, '
                  f'recon_loss: {recon_loss:.2f}, '
                  f'kl_loss: {kl_loss:.7f}',
                  f'lr: {optimizer.param_groups[0]["lr"]:.7f}')
    scheduler.step()
# Store last learning rate for fine-tuning in LDM
params['final_learning_rate'] = optimizer.param_groups[0]['lr']

# --------------------
# Evaluate predictions
# --------------------

net.eval()
X = torch.from_numpy(measured_spikes).type(torch.float32)
X_hat,mu,logvar = net(X.view(X.shape[0], -1).to(device))

# ----------
# Save Model
# ----------
model_save_path = f'{args.save_path}_prob_{args.prob_of_measurement}_sample_{args.sample_number}'
print(f'Saving model at: {model_save_path}')
torch.save(net.state_dict(), model_save_path+'.pt')
np.save(model_save_path+'_parameters.npy',params)
np.save(model_save_path+'_train_loss.npy',train_loss)
