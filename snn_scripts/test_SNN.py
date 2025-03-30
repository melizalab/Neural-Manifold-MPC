import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network_architectures import rLIF_classification as model
from stimulus_scripts.load_mnist_data import load_MNIST_data,train_val_test_split
from .SNN_diagnostics import *
from tqdm import tqdm
# -----------
# Parse Args
# -----------
p = argparse.ArgumentParser()
p.add_argument('--model_path',default='saved_models/snns/SNN_classifier')
p.add_argument('--mnist_data_path',default='mnist_data')
args = p.parse_args()

# ---------------------
# Use CUDA if available
# ---------------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'USING DEVICE: {device}')

# --------
# Load SNN
# --------
print('LOADING SNN parameters...')
model_dict = torch.load(f'{args.model_path}.pt')
# Parameters
model_params = model_dict['model_params']
model_params['is_online'] = True
# Weights
model_weights = model_dict['model_state_dict']
batch_size = 200
# ---------------
# Load MNIST data
# ---------------
print('Loading data...')
_, mnist_data = load_MNIST_data(data_path=args.mnist_data_path,download=False)
mnist_train,_,mnist_test = train_val_test_split(mnist_data,*model_params['train_val_test_ratios'],seed=model_params['training_random_seed'])
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
# --------------------------------------------------------------
# Perform multiple test trails (recall noise is added to inputs)
# --------------------------------------------------------------
n_trials = 30
train_acc = []
test_acc = []
for i,data_loader in enumerate([train_loader,test_loader]):
    with torch.no_grad():
        for trial in range(n_trials):
            snn = model.rLIF(params=model_params,return_out_V=True).to(device)
            snn.load_state_dict(clear_buffers(model_weights),strict=False)
            snn.eval()
            n_trial_obs = 0
            batch_trial_acc = 0
            for (data,labels) in tqdm(data_loader):
                data,labels = data.to(device),labels.to(device)
                _, _, out_spikes, _ = snn(data.view(len(data), -1))
                # Get classification accuracy
                n_trial_obs += len(labels)
                _,pred_labels = out_spikes.sum(dim=0).max(1)
                batch_trial_acc += (pred_labels==labels).sum().item()
            if i == 0:
                train_acc.append(batch_trial_acc/n_trial_obs)
                print(f'Trial: {trial+1}/{n_trials}, Acc: {100* train_acc[-1]:.2f}%')
            else:
                test_acc.append(batch_trial_acc/n_trial_obs)
                print(f'Trial: {trial+1}/{n_trials}, Acc: {100* test_acc[-1]:.2f}%')
train_mean = np.mean(train_acc)
train_std = np.std(train_acc)
test_mean = np.mean(test_acc)
test_std = np.std(test_acc)

