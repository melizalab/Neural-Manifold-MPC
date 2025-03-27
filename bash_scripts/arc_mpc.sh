#!/bin/bash

# Array of probabilities
probs=(.2)
arcs=(1 2)


# Loop through each probability
for prob in "${probs[@]}"
do
  # Loop through sample numbers
  for arc in "${arcs[@]}"
  do
    # loop through trials
    for j in {0..49}
    do
        echo "Running with prob_of_measurement=$prob, arc=$arc, trial=$j"
        python -m neural_manifold_control.mpc.mpc --path_to_LDM=saved_models/latent_dynamics_models/LDM_prob_0"$prob"_sample_1 --trial_id="$j" --path_to_reference_trajectory=reference_trajectories/arcs --arc_num="$arc" --path_to_save_output=neural_manifold_control/mpc/arc_"$arc"_control
    done
  done
done
