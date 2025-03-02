#!/bin/bash

# Array of probabilities
probs=(.2)


# Loop through each probability
for prob in "${probs[@]}"
do
  # Loop through sample numbers
  for i in {0..9}
  do
    # loop through trials
    for j in {0..50}
    do
        echo "Running with prob_of_measurement=$prob, sample_number=$i, trial=$j"
        python -m neural_manifold_control.mpc.mpc --path_to_LDM=saved_models/latent_dynamics_models/LDM_prob_0"$prob"_sample_"$i" --trial_id="$j"
    done
  done
done
