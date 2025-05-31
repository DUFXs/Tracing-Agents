import pandas as pd
import numpy as np
import random

# Path to the SWE-Bench Verified .parquet file
parquet_file = "swe_bench_verified.parquet"

# Load the .parquet file
df = pd.read_parquet(parquet_file)

# Set the desired total sample size
desired_samples = 100

# Calculate the proportional sample size for each stratum
stratum_sizes = df.groupby(['repo', 'difficulty']).size()
total_instances = len(df)
stratum_proportions = (stratum_sizes / total_instances) * desired_samples

# Ensure at least 1 sample per stratum
stratum_proportions = stratum_proportions.round().astype(int).clip(lower=1)

# Perform the stratified proportional sampling
sample_list = []
for (repo, difficulty), size in stratum_proportions.items():
    subset = df[(df['repo'] == repo) & (df['difficulty'] == difficulty)]
    sample_list.extend(subset.sample(n=size, random_state=42)['instance_id'].tolist())

# Print the final sample list
print(sample_list)
