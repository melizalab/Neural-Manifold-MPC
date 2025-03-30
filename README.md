# Neural-Manifold-MPC  

This repository contains the source code for implementing **Model Predictive Control (MPC) of the Neural Manifold**. The project explores how MPC can be applied to control neural activity, leveraging data-driven models of neural dynamics.  

## Features  
- **Neural Manifold Representation**: Captures the intrinsic low-dimensional structure of activity in a simulated neural circuit.  
- **Model Predictive Control (MPC)**: Implements an MPC framework to control neural dynamics toward desired states using data-driven dynamics models. 
- **Simulation and Experiments**: Includes scripts for generating simulated neural data and applying MPC and PID-based control.  

## Notes  
- Built using Python 3.8.3
- Minimial set of model examples provided to reproduce basic results of paper.
- All Pytorch models supplied were trained using CUDA.

## Basic Usage  
1. Clone the repository:  
   ```bash
   git clone https://github.com/melizalab/Neural-Manifold-MPC.git
   cd Neural-Manifold-MPC
   ```  
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run MPC for a single trial:  
   ```bash
   python -m neural_manifold_control.mpc.mpc
   ```  
# How to regenerate all data from paper

## Construct Stimulus VAE (sVAE)
If running for first time and MNIST dataset has not be downloaded:
```
python -m stimulus_scripts.train_stimulus_VAE --download_MNIST
```
Remove ```--download_MNIST``` flag if MNIST already in folder.

Can test and visualize results with:
```
python -m stimulus_scripts.test_stimulus_VAE
```

## Train Artificial Circuit (AC)
If running without having downloading MNIST as above, run:
```
python -m snn_scripts.train_SNN --download_MNIST
```
Remove ```--download_MNIST``` flag if MNIST already in folder.

Can test results and build distribution of classification accuracy with:
```
python -m snn_scripts.test_SNN
```

## Construct Neuron VAE (nVAE)

## Construct Latent Dynamics Model (LDM)

## Get Reference Trajectories

## Perform MPC

## Perform PID Control