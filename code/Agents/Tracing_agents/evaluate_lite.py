import os
from pathlib import Path

def evaluate_reproduction(dataset_base_path):
    # Initialize counters
    total_default = 0
    total_refined = 0
    total_issues = 0

    # Ensure the base path exists
    base_path = Path(dataset_base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory '{dataset_base_path}' not found.")

    # Iterate through each issue directory
    for issue_dir in base_path.glob("SWE-bench_Lite_*"):
        if issue_dir.is_dir():
            total_issues += 1
            issue_name = issue_dir.name
            default_file = issue_dir / "reproduction_test_samples_temp_0.8_1_samples_gpt-4o-mini-2024-07-18" /"output_0_processed_reproduction_test_verified.jsonl"

            # (1) Check default reproduction
            default_reproduced = 0
            if default_file.exists() and os.path.getsize(default_file) > 0:
                default_reproduced = 1
                total_default += 1

            # (2) Check refined reproduction
            refined_reproduced = 0
            for refined_folder in issue_dir.glob("reproduction_test_samples_temp_0.8_1_samples_gpt-4o-mini-2024-07-18_refined_*"):
                refined_file = refined_folder / "output_0_processed_reproduction_test_verified.jsonl"
                if refined_file.exists() and os.path.getsize(refined_file) > 0:
                    refined_reproduced = 1
                    total_refined += 1
                    break  # Count as reproduced if any refined folder is non-empty
            
            print(f"Issue: {issue_name}, Default: {default_reproduced}, Refined: {refined_reproduced}")

    # Final Statistics
    if total_issues == 0:
        print("No valid issue directories found. Please check the dataset base path.")
    else:
        print("\nFinal Statistics:")
        print(f"Default Reproduced: {total_default} / {total_issues} ({(total_default / total_issues) * 100:.2f}%)")
        print(f"Refined Reproduced: {total_refined} / {total_issues} ({(total_refined / total_issues) * 100:.2f}%)")

if __name__ == "__main__":
    # Set the correct base directory
    dataset_base_path = "/Users/ic/Desktop/Agents/Agentless/results/princeton-nlp"
    evaluate_reproduction(dataset_base_path)
