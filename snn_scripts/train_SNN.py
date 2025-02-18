import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from network_architectures import rLIF_classification as model
from stimulus_scripts.load_mnist_data import load_MNIST_data,train_val_test_split
from tqdm import tqdm
from .SNN_diagnostics import *
import copy

# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
# Path to Save SNN
p.add_argument('--model_name',default='SNN_classifier',type=str)
p.add_argument("--save_path", default='saved_models/snns',type=str)
# SNN Structure Hyperparameters
p.add_argument("--sensory_layer_size", default=100, type=int)
p.add_argument("--reservoir_layer_size", default=500, type=int)
p.add_argument('--decay_rate',default=0.99,type=float)
p.add_argument("--beta", default=0.99, type=float)
p.add_argument("--noise_scaling",default=0.5,type=float)
p.add_argument('--clamp_value',default=-2.0,type=str)
# Training hyperparameters
p.add_argument('--is_online',default=False,type=bool)
p.add_argument("--stimulus_n_steps", default=50, type=int)
p.add_argument("--batch_size", default=128, type=int)
p.add_argument("--num_epochs", default=100, type=int)
p.add_argument('--learning_rate',default=5e-4,type=float)
p.add_argument('--train_val_test_ratios',nargs='+',default=[0.7,0.15,0.15])
p.add_argument('--early_stopping_threshold',default=5, type = int)
p.add_argument('--training_random_seed',default=138,type=int)
# MNIST data paths
p.add_argument('--mnist_data_path',default='mnist_data')
p.add_argument('--download_MNIST',default=False)
args = p.parse_args()

# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')

# ------------------------
# SNN Parameter Dictionary
# ------------------------
model_params = {}
model_params['beta'] = args.beta
model_params['sensory_layer_size'] = args.sensory_layer_size
model_params['reservoir_layer_size'] = args.reservoir_layer_size
model_params['stimulus_n_steps'] = args.stimulus_n_steps
model_params['noise_scaling'] = args.noise_scaling
model_params['clamp_value'] = args.clamp_value
model_params['batch_size'] = args.batch_size
model_params['num_epochs'] = args.num_epochs
model_params['learning_rate'] = args.learning_rate
model_params['is_online'] = args.is_online
model_params['decay_rate'] = args.decay_rate
model_params['train_val_test_ratios'] = args.train_val_test_ratios
model_params['training_random_seed'] = args.training_random_seed

# ------------------------
# Load MNIST training data
# ------------------------
print('LOADING TRAINING DATA...')
_, mnist_data = load_MNIST_data(data_path=args.mnist_data_path,download=args.download_MNIST)
mnist_train,mnist_val,_ = train_val_test_split(mnist_data,*args.train_val_test_ratios,seed=args.training_random_seed)
train_loader = DataLoader(mnist_train,batch_size=args.batch_size,shuffle=True,drop_last=False)
val_loader = DataLoader(mnist_val,batch_size=args.batch_size,shuffle=False,drop_last=False)

# ------------
# Instance SNN
# ------------
print('BUILDING NETWORK...')
model = model.rLIF(params=model_params).to(device)

# ---------------------------
# Loss Function and Optimizer
# ---------------------------
print('INITIALIZING LOSS AND OPTIMIZER...')
loss_ce = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
lmbda = lambda epoch: args.decay_rate**epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)

# -------------
# Training Loop
# -------------
best_val_acc = float('-inf')
train_loss = []
train_acc = []
val_loss = []
val_acc = []
print('Training SNN...')
for epoch in range(args.num_epochs):
    model.train()
    epoch_train_acc = 0
    n_train_obs = 0
    epoch_val_acc = 0
    n_val_obs = 0
    for (data, labels) in tqdm(train_loader):
        # Get training data
        data = data.to(device)
        labels = labels.to(device)

        # Forward pass where each stimulus is presented for stimulus_n_steps
        sense_spikes, res_spikes, out_spikes, out_V = model(data.view(len(data), -1))

        # Initialize loss and sum over time of stimulus
        batch_loss = torch.zeros(1, dtype=torch.float, device=device)
        for step in range(args.stimulus_n_steps):
            batch_loss += loss_ce(out_V[step], labels)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # Store training loss
        train_loss.append(batch_loss.detach().item())

        # Get classification accuracy
        n_train_obs += len(labels)
        _,pred_labels = out_spikes.sum(dim=0).max(1)
        epoch_train_acc += (pred_labels==labels).sum().item()
    train_acc.append(epoch_train_acc/n_train_obs)


    # Compute validation accuracy
    model.eval()
    with torch.no_grad():
        for (val_data, val_labels) in tqdm(val_loader):
            val_data = val_data.to(device)
            val_labels = val_labels.to(device)
            _, _, out_val_spikes, _ = model(val_data.view(len(val_data), -1))
            # Get classification accuracy
            n_val_obs += len(val_labels)
            _,pred_val_labels = out_val_spikes.sum(dim=0).max(1)
            epoch_val_acc += (pred_val_labels==val_labels).sum().item()
    val_acc.append(epoch_val_acc/n_val_obs)

    # Number of spikes
    num_sense_spikes = int(torch.sum(sense_spikes).item())
    num_res_spikes = int(torch.sum(res_spikes).item())
    num_out_spikes = int(torch.sum(out_spikes).item())

    print(f'EPOCH: {epoch+1}/{args.num_epochs}, '
          f'Training Acc: {train_acc[-1]:.2f}, '
          f'Validation Acc: {val_acc[-1]:.2f}, '
          f'Number of Spikes (Sen., Res., Out.): {num_sense_spikes}, {num_res_spikes}, {num_out_spikes}')
    
    # Check for best model
    if val_acc[-1] > best_val_acc:
        print(f"New best model found at epoch {epoch + 1}!")
        best_val_acc = val_acc[-1]
        early_stop_counter = 0 
        # Save net
        model_params['stopped_at_epoch'] = epoch+1
        torch.save({'model_params':model_params,
                    'last_train_acc':train_acc[-1],
                    'last_val_acc':val_acc[-1],
                    'model_state_dict': copy.deepcopy(model.state_dict())},
                    f'{args.save_path}/{args.model_name}.pt')
    else:
        early_stop_counter += 1
        print(f"No improvement of validation loss in {early_stop_counter} epoch(s).")

    # Early stopping check
    if early_stop_counter >= args.early_stopping_threshold:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break
    
    scheduler.step()
