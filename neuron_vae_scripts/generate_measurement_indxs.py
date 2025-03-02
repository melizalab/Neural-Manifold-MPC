import numpy as np
import argparse
from .sample_indxs import get_sample
import pandas as pd

# ----------
# Parse Args
# ----------
p = argparse.ArgumentParser()
p.add_argument("--out_path", default="assimilation_data/spikes_measurement_indxs")
p.add_argument('--n_samples_per_prob', default=10, type=int)
args = p.parse_args()

# List of probabilities
probs = [.01, .05, .10, .20, .30, .40, .50, .60]

# Initialize data and index storage
indxs = []
prob_sample = []

for prob in probs:
    for sample in range(args.n_samples_per_prob):
        # Generate sample data (assuming get_sample returns a dict)
        measurement_indxs = get_sample(prob)

        # Append data and index
        indxs.append(measurement_indxs)
        prob_sample.append((prob, sample))

# Create MultiIndex
multi_index = pd.MultiIndex.from_tuples(prob_sample, names=["prob", "sample"])

# Create DataFrame without specifying columns
measurement_indx_df = pd.DataFrame(indxs, index=multi_index)

# Save DataFrame to the specified path
measurement_indx_df.to_pickle(f'{args.out_path}.pkl')

# Print the DataFrame for verification
print(measurement_indx_df)