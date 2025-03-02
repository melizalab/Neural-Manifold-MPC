#!/bin/bash

# Array of probabilities
probs=(.2)

# Loop through each probability
for prob in "${probs[@]}"
do
  # Loop through sample numbers
  for i in {0..9}
  do
    echo "Running with prob_of_measurement=$prob and sample_number=$i"
    python -m reference_trajectories.generate_set_point_reference_trajectories --prob_of_measurement="$prob" --sample_number="$i"
  done
done
