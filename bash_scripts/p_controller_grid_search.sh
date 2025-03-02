#!/bin/bash

# Define the range of p_gains values
declare -a p_gains_list=(0 0.01 0.1 0.5 1 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)


# Loop through all combinations of p_gains
for p1 in "${p_gains_list[@]}"; do
    for p2 in "${p_gains_list[@]}"; do
        echo "Running with p_gains=[$p1, $p2]"
        python -m neural_manifold_control.reactive_control.reactive_control --p_gains $p1 $p2
    done
done