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
# In Depth Workflow of Neural Manifold Control

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
1. Generate data used to stimulate AC (V assimilation data):
```
python -m stimulus_scripts.generate_stimulus_assimilation_data
```
2. Stimuluate AC with V assimilation:
```
python -m neuron_vae_scripts.generate_spikes_for_vae
```
3. Filter spikes with exponentially weighted lowpass filter (EWMA):
```
python -m neuron_vae_scripts.filter_spikes_with_ewma
```
4. For each levels of observation percetange, we want 10 randomly drawn ensembles for the AC. To get the indexes that responds to measuring specific neurons in the AC, run:
```
python -m neuron_vae_scripts.generate_measurement_indxs
```
Note that the set of measured indexes used in the paper is provided in the ```assimiliation_data/spikes_measurement_indxs.pkl``` file in the repo.

5. Pretrain the nVAE:
```
python -m neuron_vae_scripts.pretrain_SNN_VAE
```

## Construct Latent Dynamics Model (LDM)
Train LDM model using sVAE and pretrained nVAE:
```
python -m latent_dynamics_model.train_latent_dynamics_model
```

Test LDM and save resulting latent state predictions (Z assimilation):
```
python -m latent_dynamics_model.test_latent_dynamics_model
```

## Get Reference Trajectories
Generate set point reference trajectories (for experiments I and II):
```
python -m reference_trajectories.generate_set_point_reference_trajectories
```
Generate arc refernece trajectories (for experiment III):
```
python -m refernce_trajectories.generate_arc_reference_trajectories
```

## Perform MPC
Run MPC for set point reference trajectories:
```
python -m neural_manifold_control.mpc.mpc
```
Run MPC for arc reference trajectories:
```
python -m neural_manifold_control.mpc.mpc --path_to_reference_trajectory=reference_trajectories/arcs --arc_num=1 --path_to_save_output=neural_manifold_control/mpc/arc_1_control
```
Make corresponding changes in args for ```arc_num=2```.

## Perform PID Control
Run PID for set point reference trajectories:
```
python -m neural_manifold_control.reactive_control.reactive_control
```

## Bash Scripts and Figures
In paper, many trials were performed for each ensemble. The trial id is controlled by changing the ```--trail_id=number``` arg in the control scripts.

After sVAE is trained and AC spikes have been filtered, run the following scripts to automate pretraining the nVAEs, LDM training/testing, reference trajectory generation, and MPC/PID control.
```
bash_scripts/pretrain_SNN_VAE.sh
bash_scripts/train_latent_dynamics_model.sh
bash_scripts/test_latent_dynamics_model.sh
bash_scripts/generate_set_point_reference_trajectories.sh
bash_scripts/generate_arc_referenence_trajectories.sh
bash_scripts/set_point_mpc.sh
bash_scripts/set_point_pid_control.sh
bash_scripts/arc_mpc.sh
```

Figures in the manuscript can be recreated with their corresponding script in ```figures/scripts``` folder.
E.g.:
```
python -m figures.scripts.Figure_2_A
```