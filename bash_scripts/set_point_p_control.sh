#!/bin/bash

# Array of probabilities
probs=(.05 .1 .3 .4 .5 .6)


# Loop through each probability
for prob in "${probs[@]}"
do
  # Loop through sample numbers
  for i in {0..9}
  do
    # loop through trials
    for j in {0..19}
    do
      j_offset=$((j + 30)) 
        echo "Running with prob_of_measurement=$prob, sample_number=$i, trial=$j_offset"
        python -m neural_manifold_control.reactive_control.reactive_control --path_to_LDM=saved_models/latent_dynamics_models/LDM_prob_0"$prob"_sample_"$i" --trial_id="$j_offset"
    done
  done
done
