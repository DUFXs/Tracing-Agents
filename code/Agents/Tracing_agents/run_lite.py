import os
import pandas as pd
import subprocess

# Path to the parquet file (you may need to download it first)
parquet_file = "swe_bench_lite.parquet"

# Read the parquet file
df = pd.read_parquet(parquet_file)

# Ensure the instance_id column exists
if "instance_id" not in df.columns:
    raise ValueError("The 'instance_id' column is missing from the parquet file.")

# Base directory for results
base_dir = os.path.realpath("results/princeton-nlp")

# Iterate over each instance_id and run the command
for instance_id in df["instance_id"].dropna().unique():
    
    # Construct the real path to the expected output file
    output_file = os.path.join(
        base_dir,
        f"SWE-bench_Lite_{instance_id}",
        "reproduction_test_samples_temp_0.8_1_samples_gpt-4o-mini-2024-07-18_refined_3-3",
        "output_0_processed_reproduction_test_verified.jsonl"
    )
    
    print("CHECKING!!!!!..........", output_file)
    # Skip if the output file already exists
    if os.path.exists(output_file):
        print("SKIPPING!!!!!..........", output_file)
        print(f"Skipping {instance_id} - Output file already exists at {output_file}.")
        continue
    
    # Run the reproduction test command
    command = [
        "python",
        "run_repro_tests.py",
        str(instance_id),
        "claude-3-5-sonnet-20241022",
        "princeton-nlp/SWE-bench_Lite",
        "-n",
        "1"
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)
