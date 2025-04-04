#!/bin/bash

# Array of probabilities
probs=(.01 .05 .1 .2 .3 .4 .5 .6)

# Loop through each probability
for prob in "${probs[@]}"
do
  # Loop through sample numbers
  for i in {0..9}
  do
    echo "Running with prob_of_measurement=$prob and sample_number=$i"
    python -m latent_dynamics_model.test_latent_dynamics_model --model_path=saved_models/latent_dynamics_models/LDM_prob_0"$prob"_sample_"$i"
  done
done
