import os
import pandas as pd
import subprocess

# Path to the parquet file (you may need to download it first)
parquet_file = "swe_bench_verified.parquet"

# Read the parquet file
df = pd.read_parquet(parquet_file)

# Ensure the instance_id column exists
if "instance_id" not in df.columns:
    raise ValueError("The 'instance_id' column is missing from the parquet file.")

# Base directory for results
base_dir = os.path.realpath("results/princeton-nlp")

stratified = ['astropy__astropy-14539', 'django__django-10554', 'django__django-14631', 'django__django-13344', 'django__django-11138', 'django__django-13346', 'django__django-11149', 'django__django-13786', 'django__django-13315', 'django__django-11333', 'django__django-15382', 'django__django-14559', 'django__django-15916', 'django__django-13028', 'django__django-11477', 'django__django-13195', 'django__django-12262', 'django__django-15973', 'django__django-16502', 'django__django-11790', 'django__django-10973', 'django__django-14122', 'django__django-16901', 'django__django-15022', 'django__django-16454', 'django__django-14349', 'django__django-15563', 'django__django-17084', 'django__django-13741', 'django__django-12193', 'django__django-14752', 'django__django-15741', 'django__django-10097', 'django__django-12308', 'django__django-13551', 'django__django-15380', 'django__django-11239', 'django__django-13933', 'django__django-16429', 'django__django-13406', 'django__django-17029', 'django__django-15104', 'django__django-11433', 'django__django-11066', 'django__django-11880', 'django__django-12741', 'matplotlib__matplotlib-14623', 'matplotlib__matplotlib-22871', 'matplotlib__matplotlib-24637', 'matplotlib__matplotlib-20488', 'matplotlib__matplotlib-25287', 'matplotlib__matplotlib-25332', 'matplotlib__matplotlib-13989', 'mwaskom__seaborn-3187', 'pallets__flask-5014', 'psf__requests-6028', 'psf__requests-1142', 'pydata__xarray-3993', 'pydata__xarray-6721', 'pydata__xarray-6938', 'pydata__xarray-2905', 'pydata__xarray-4094', 'pydata__xarray-6992', 'pylint-dev__pylint-8898', 'pylint-dev__pylint-4661', 'pylint-dev__pylint-4970', 'pytest-dev__pytest-10356', 'pytest-dev__pytest-5631', 'pytest-dev__pytest-7490', 'pytest-dev__pytest-5262', 'pytest-dev__pytest-7432', 'scikit-learn__scikit-learn-25102', 'scikit-learn__scikit-learn-10297', 'scikit-learn__scikit-learn-10844', 'scikit-learn__scikit-learn-14087', 'scikit-learn__scikit-learn-12973', 'scikit-learn__scikit-learn-25232', 'scikit-learn__scikit-learn-14496', 'scikit-learn__scikit-learn-11310', 'sphinx-doc__sphinx-8548', 'sphinx-doc__sphinx-10466', 'sphinx-doc__sphinx-10614', 'sphinx-doc__sphinx-7757', 'sphinx-doc__sphinx-10323', 'sphinx-doc__sphinx-8721', 'sphinx-doc__sphinx-8269', 'sphinx-doc__sphinx-10435', 'sphinx-doc__sphinx-7590', 'sympy__sympy-12489', 'sympy__sympy-23824', 'sympy__sympy-20428', 'sympy__sympy-20438', 'sympy__sympy-23413', 'sympy__sympy-22914', 'sympy__sympy-24066', 'sympy__sympy-13551', 'sympy__sympy-14976', 'sympy__sympy-13798', 'sympy__sympy-15875', 'sympy__sympy-19495', 'sympy__sympy-12096', 'sympy__sympy-24539', 'sympy__sympy-16886', 'sympy__sympy-13878']

# Iterate over each instance_id and run the command
for instance_id in stratified:
    
    # # Proceed only if the instance_id starts with "psf"
    # if not instance_id.startswith("psf"):
    #     print(f"Skipping {instance_id} - Does not start with 'psf'.")
    #     continue
    
    # Construct the real path to the expected output file
    output_file = os.path.join(
        base_dir,
        f"SWE-bench_Verified_{instance_id}",
        "reproduction_test_samples_temp_0.8_30_samples_claude-3-5-sonnet-20241022_refined_3-3",
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
        "princeton-nlp/SWE-bench_Verified",
        "-n",
        "30"
    ]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)
