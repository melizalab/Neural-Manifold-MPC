#!/bin/bash

# Array of probabilities
probs=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6)

# Loop through each probability
for prob in "${probs[@]}"
do
  # Loop through sample numbers
  for i in {0..9}
  do
    echo "Running with prob_of_measurement=$prob and sample_number=$i"
    python -m figures.scripts.forecast_spikes --prob_of_measurement="$prob" --sample_number="$i"
  done
done
