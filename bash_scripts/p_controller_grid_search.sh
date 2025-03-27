#!/bin/bash

# Define the range of p_gains values
declare -a p_gains_list=(0 0.1 1 10 20 50)

# Loop through all combinations of p_gains
for k11 in "${p_gains_list[@]}"; do
    for k12 in "${p_gains_list[@]}"; do
        for k21 in "${p_gains_list[@]}"; do
            for k22 in "${p_gains_list[@]}"; do 
                echo "Running with p_gains=[$k11, $k12, $k21, $k22]"
                python -m neural_manifold_control.reactive_control.reactive_control --p_gains $k11 $k12 $k21 $k22
            done
        done
    done
done
