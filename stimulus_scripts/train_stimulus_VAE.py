import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import copy
import torch.nn as nn

# ------------------------------
# Append path for custom modules
# ------------------------------
from network_architectures import CVAE as model
from .load_mnist_data import load_MNIST_data,train_val_test_split

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
# Model hyperparameters
p.add_argument('--model_name', default='MNIST_VAE', type=str)
p.add_argument('--latent_dim_size', default=2, type=int)
p.add_argument("--save_path", default='saved_models/stimulus_vaes')
# MNIST data paths
p.add_argument('--mnist_data_path',default='mnist_data')
p.add_argument('--download_MNIST',default=False)
# Training hyperparameters
p.add_argument('--kl_hyperparameters',nargs='+',default=[1,1],type = float)
p.add_argument('--train_val_test_ratios',nargs='+',default=[0.7,0.15,0.15])
p.add_argument('--random_seed',default=None,type=int)
p.add_argument('--batch_size',default=128, type=int)
p.add_argument('--learning_rate', default=5e-4, type=float)
p.add_argument("--max_num_epochs", default=100, type=int)
p.add_argument('--early_stopping_threshold',default=5, type = int)
args = p.parse_args()


# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')


# ------------------------
# Load MNIST training data
# ------------------------
print('LOADING DATA...')
mnist_data,_ = load_MNIST_data(data_path=args.mnist_data_path,download=args.download_MNIST)
mnist_train,mnist_val,_ = train_val_test_split(mnist_data,*args.train_val_test_ratios,seed=args.random_seed)
train_loader = DataLoader(dataset=mnist_train,batch_size=args.batch_size,shuffle=True,drop_last=True)
val_loader = DataLoader(dataset=mnist_val,batch_size=args.batch_size,shuffle=True,drop_last=True)


# -------------------------
# CVAE Parameter Dictionary
# -------------------------
params = {'image_size':784,
          'latent_dim_size': args.latent_dim_size,
          'kl_hyperparameters': args.kl_hyperparameters,
          'learning_rate': args.learning_rate,
          'batch_size': args.batch_size,
          'num_epochs': args.max_num_epochs,
          'train_val_test_ratios':args.train_val_test_ratios,
          'training_random_seed':args.random_seed}


# -------------
# Instance CVAE
# -------------
print('BUILDING NETWORK...')
model = model.VAE(params).to(device)

# --------------------
# Initialize optimizer
# --------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
def VAE_loss(kl_hyperparameters,predicted_x,true_x,mu,logvar):
    MSE = nn.MSELoss(reduction='sum')
    recon_loss = MSE(true_x,predicted_x)
    kl_loss = -0.5*torch.sum(1+logvar-mu**2-logvar.exp())
    total_loss = kl_hyperparameters[0]*recon_loss+kl_hyperparameters[1]*kl_loss
    return total_loss,recon_loss,kl_loss


# -------------------------------------
# Function to check validation accuracy
# -------------------------------------
def validation_loss(net,val_loader):
    val_loss, val_obs = 0,0
    net.eval()
    with torch.inference_mode():
        for (images, _) in tqdm(val_loader):
            X = images.to(device)
            X_hat, mu, logvar = net(X)
            val_batch_loss,_,_ = VAE_loss(kl_hyperparameters=args.kl_hyperparameters,
                                          predicted_x=X_hat.reshape(args.batch_size,784),
                                          true_x=X.reshape(args.batch_size,-1),
                                          mu=mu,
                                          logvar=logvar)
            val_loss+=val_batch_loss.item()
            val_obs+= len(images)
    mean_val_loss = val_loss/val_obs
    return mean_val_loss

# -----------
# Train Model
# -----------
print('TRAINING MODEL...')
best_val_loss = float('inf')
train_loss = np.zeros(args.max_num_epochs)
val_loss = np.zeros(args.max_num_epochs)
for epoch in range(args.max_num_epochs):
    model.train()
    epoch_training_loss = 0
    train_obs = 0
    # Get only images since doing a reconstruction task (labels not needed)
    for (images,_) in tqdm(train_loader):
        optimizer.zero_grad()
        
        # Sample batch and forward pass through CVAE
        X = images.to(device)
        X_hat,mu,logvar = model(X)
        
        # Calculate loss and update weights
        batch_loss,recon_loss,kl_loss = VAE_loss(kl_hyperparameters=args.kl_hyperparameters,
                                                 predicted_x=X_hat.reshape(args.batch_size,784),
                                                 true_x=X.reshape(args.batch_size,-1),
                                                 mu=mu,
                                                 logvar=logvar)
        epoch_training_loss += batch_loss.item()
        train_obs += len(images)
        batch_loss.backward()
        optimizer.step()
    
    # Get mean loss for training and validation data at each epoch
    mean_train_loss = epoch_training_loss/train_obs
    mean_val_loss = validation_loss(model,val_loader)
    
    # Get mean losses and print for monitoring
    train_loss[epoch] = mean_train_loss
    val_loss[epoch] = mean_val_loss
    print(f'Epoch: {epoch + 1}/{args.max_num_epochs}, '
          f'Mean Training Loss: {train_loss[epoch]:.2f}, '
          f'Mean Validation Loss: {val_loss[epoch]:.2f}')

    # Check for best model
    if mean_val_loss < best_val_loss:
        print(f"New best model found at epoch {epoch + 1}!")
        best_val_loss = mean_val_loss
        early_stop_counter = 0 
        # Save net
        params['stopped_at_epoch'] = epoch+1
        torch.save({'model_params':params,
                    'last_train_loss':train_loss[epoch],
                    'last_val_loss':val_loss[epoch],
                    'model_state_dict': copy.deepcopy(model.state_dict())},
                    f'{args.save_path}/{args.model_name}.pt')
    else:
        early_stop_counter += 1
        print(f"No improvement of validation loss in {early_stop_counter} epoch(s).")

    # Early stopping check
    if early_stop_counter >= args.early_stopping_threshold:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break

# -------------------------
# Plot losses if applicable
# -------------------------
plt.plot(train_loss,alpha = 0.7,label='train')
plt.plot(val_loss,alpha = 0.7,label='val')
plt.ylabel('Average Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

breakpoint()