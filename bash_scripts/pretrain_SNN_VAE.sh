#!/bin/bash

# Array of probabilities
probs=(.2)
# Loop through each probability
for prob in "${probs[@]}"
do
  for i in {0..9}
  do
    echo "Running with prob_of_measurement=$prob and sample_number=$i"
    python -m neuron_vae_scripts.pretrain_SNN_VAE --prob_of_measurement="$prob" --sample_number="$i"
  done
done