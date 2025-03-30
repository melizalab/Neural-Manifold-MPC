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

## Train Artificial Circuit (AC)

## Construct Neuron VAE (nVAE)

## Construct Latent Dynamics Model (LDM)

## Get Reference Trajectories

## 