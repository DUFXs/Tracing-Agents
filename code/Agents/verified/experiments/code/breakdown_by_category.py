#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import re


def extract_issue_id(filename: str) -> str:
    """
    Extracts the issue ID from a JSON filename.
    Expected filename pattern:
    clone_detection_result_<issue_id>_<timestamp>_... .json
    Where issue_id format is project__project-number (no underscores inside project or number).
    """
    match = re.match(r'^clone_detection_result_([^_]+__[^_]+)_', filename)
    return match.group(1) if match else None


def main():
    parser = argparse.ArgumentParser(
        description="Map issue IDs from JSON filenames to difficulty levels from a Parquet file and count occurrences."
    )
    parser.add_argument(
        "json_folder",
        type=str,
        help="Path to the folder containing JSON files."
    )
    parser.add_argument(
        "parquet_file",
        type=str,
        help="Path to the Parquet file with 'instance_id' and 'difficulty' columns."
    )
    args = parser.parse_args()

    folder = Path(args.json_folder)
    if not folder.is_dir():
        raise ValueError(f"Provided JSON folder path is not a directory: {folder}")

    # Load Parquet file into a DataFrame
    df = pd.read_parquet(args.parquet_file)
    
    # Ensure required columns exist
    required_cols = {'instance_id', 'difficulty'}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Parquet file must contain columns: {required_cols}, found: {df.columns.tolist()}"
        )

    # Build a mapping from instance_id to difficulty
    difficulty_map = dict(zip(df['instance_id'], df['difficulty']))

    # Precompute total counts per difficulty in the Parquet
    total_per_difficulty = df['difficulty'].value_counts().to_dict()

    # Counters for difficulties and missing mappings
    counts = {}
    missing_ids = []
    total_processed = 0

    # Iterate over JSON files and map to difficulty
    for json_file in folder.glob("*.json"):
        issue_id = extract_issue_id(json_file.name)
        if not issue_id:
            print(f"Skipping file with unrecognized format: {json_file.name}")
            continue
        total_processed += 1
        difficulty = difficulty_map.get(issue_id)
        if difficulty is None:
            missing_ids.append(issue_id)
        else:
            counts[difficulty] = counts.get(difficulty, 0) + 1

    # Output the breakdown
    print("\nIssue counts per difficulty category (processed/total in Parquet, percentage solved):")
    # Ensure we list all categories present in Parquet
    for difficulty in sorted(total_per_difficulty):
        processed = counts.get(difficulty, 0)
        total = total_per_difficulty[difficulty]
        percentage = (processed / total * 100) if total > 0 else 0
        print(f"- {difficulty}: {processed}/{total} ({percentage:.2f}%)")

    mapped_count = sum(counts.values())
    print(f"\nTotal mapped: {mapped_count}/{total_processed} issues processed.")

    # Warn about missing IDs
    if missing_ids:
        missing_unique = sorted(set(missing_ids))
        print(f"\nWarning: {len(missing_unique)} issue IDs were not found and were skipped:")
        for mid in missing_unique:
            print(f"  - {mid}")


if __name__ == "__main__":
    main()
