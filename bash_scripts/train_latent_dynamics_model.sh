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
    python -m latent_dynamics_model.train_latent_dynamics_model --prob_of_measurement="$prob" --sample_number="$i"
  done
done
