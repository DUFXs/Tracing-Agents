import os
import json
import pandas as pd
import re
from collections import defaultdict
import black

def clean_patch_string(patch_text):
    """
    Cleans patch text: splits modifications by file, removes diff metadata, keeps added & unchanged lines.
    """
    lines = patch_text.splitlines()
    file_patches = defaultdict(list)  # Store modifications per file
    current_file = None

    for line in lines:
        # Identify file modification
        match = re.match(r"^diff --git a/(\S+) b/(\S+)", line)
        if match:
            current_file = match.group(2)
            continue  

        if line.startswith(("@@", "---", "+++")):
            continue  # Skip diff metadata

        if line.startswith("+") and current_file:
            file_patches[current_file].append(line[1:].lstrip())  # Remove '+' sign
        elif not line.startswith("-") and current_file:  # Keep unchanged lines
            file_patches[current_file].append(line)

    # Join cleaned lines per file
    formatted_patches = {}
    for file, lines in file_patches.items():
        code = "\n".join(lines) + "\n"
        if file.endswith(".py"):
            try:
                code = black.format_str(code, mode=black.Mode())  # Format with black
            except:
                pass  
        formatted_patches[file] = code

    return formatted_patches

# Load Parquet file
parquet_file = "swe-bench-verified.parquet"
df = pd.read_parquet(parquet_file, engine="pyarrow")

# Dictionary to store structured data
results = defaultdict(lambda: defaultdict(dict))

# List of models to evaluate
# models = [
#     "20250110_learn_by_interact_claude3.5",
#     "20250110_blackboxai_agent_v1.1",
#     "20250117_wandb_programmer_o1_crosscheck5",
#     "20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022",
#     "20241221_codestory_midwit_claude-3-5-sonnet_swe-search",
#     "20240612_MASAI_gpt4o",
#     "20240620_sweagent_claude3.5sonnet",
#     "20241202_agentless-1.5_claude-3.5-sonnet-20241022",
#     "20241029_OpenHands-CodeAct-2.1-sonnet-20241022"
# ]

models = [
    "20241108_autocoderover-v2.0-claude-3-5-sonnet-20241022",
    "20240612_MASAI_gpt4o",
    "20240620_sweagent_claude3.5sonnet",
    "20241202_agentless-1.5_claude-3.5-sonnet-20241022",
    "20241028_agentless-1.5_gpt4o",
    "20241029_OpenHands-CodeAct-2.1-sonnet-20241022",
    "20250122_autocoderover-v2.1-claude-3-5-sonnet-20241022"
]

# Base path where evaluation logs are stored
base_path = "/Users/ic/Desktop/Agents/SWE-bench-folder/experiments/evaluation/verified"

# Process ground truth patches and organize instance_ids by difficulty
for _, row in df.iterrows():
    instance_id = row["instance_id"]
    difficulty = row["difficulty"]
    raw_patch = row.get("patch", "")  # Ground truth patch from parquet
    cleaned_ground_truth_patch = clean_patch_string(raw_patch)  # Apply cleaning function

    # Iterate over models
    for agent in models:
        report_path = os.path.join(base_path, agent, "logs", instance_id, "report.json")
        patch_path = os.path.join(base_path, agent, "logs", instance_id, "patch.diff")

        # Default values
        resolved = None
        cleaned_agent_patch = {}

        # Load resolution status from report.json
        if os.path.exists(report_path):
            try:
                with open(report_path, "r") as f:
                    report_data = json.load(f)
                resolved = report_data.get(instance_id, {}).get("resolved", None)
            except:
                pass  # Handle missing or unreadable report

        # Load agent patch from patch.diff and clean
        if os.path.exists(patch_path):
            try:
                with open(patch_path, "r") as f:
                    raw_agent_patch = f.read()
                cleaned_agent_patch = clean_patch_string(raw_agent_patch)
            except:
                pass  # Handle missing or unreadable patch

        # Store results in dictionary
        results[difficulty][instance_id][agent] = {
            "resolved": resolved,
            "agent_patch": cleaned_agent_patch,
            "ground_truth_patch": cleaned_ground_truth_patch
        }

output_file = "evaluation_results_taxonomy_agents.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file}")
