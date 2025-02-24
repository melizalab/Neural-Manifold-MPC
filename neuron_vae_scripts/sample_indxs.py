import numpy as np

def get_sample(prob_of_obs):

    # Get Indexes to Measure from SNN
    # (100 neurons in sensory, 500 neurons in reservoir, 10 neurons in output layer)
    n_neurons = 610
    n_measurements = int(n_neurons*prob_of_obs)
    indxs = np.random.choice(range(610), replace=False, size=int(n_measurements))

    # Convert to corresponding indxs in the sensory, reservoir, and output layers of the SNN
    sensory_indxs = indxs[indxs<100]
    reservoir_indxs = indxs[np.where((indxs>99) & (indxs <600))] - 100
    output_indxs = indxs[indxs>599] - 600

    # Save indexes
    measurement_indxs = {'raw_indxs': indxs,
                         'sensory_indxs':sensory_indxs,
                         'reservoir_indxs':reservoir_indxs,
                         'output_indxs':output_indxs}
    return measurement_indxs
