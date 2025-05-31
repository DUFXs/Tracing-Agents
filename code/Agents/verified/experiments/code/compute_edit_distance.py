#!/usr/bin/env python3
"""
compute_edit_distance.py

Compute the Levenshtein edit distance between the agent patch and the ground-truth
patch for every JSON file in a folder.

Usage
-----
    python compute_edit_distance.py /path/to/folder
"""

import argparse
import json
import pathlib
import statistics
from typing import Dict, List, Tuple

###############################################################################
# Small, memory-efficient Levenshtein implementation (O(min(n,m)) extra space)
###############################################################################
def levenshtein(a: str, b: str) -> int:
    """Return the character-level Levenshtein distance between strings *a* and *b*."""
    if a == b:
        return 0
    # Make sure len(a) >= len(b) to use less memory.
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            insert_cost = previous[j] + 1
            delete_cost = current[j - 1] + 1
            subst_cost  = previous[j - 1] + (ca != cb)
            current.append(min(insert_cost, delete_cost, subst_cost))
        previous = current
    return previous[-1]

###############################################################################
# Patch-selection logic prescribed in the prompt
###############################################################################
def choose_patch_strings(agent_patch: Dict[str, str],
                         gt_patch: Dict[str, str]) -> Tuple[str, str, List[str]]:
    """Return the two strings whose edit distance we should compute, plus the list
    of file names that were actually compared."""
    agent_files = set(agent_patch)
    gt_files    = set(gt_patch)
    shared      = agent_files & gt_files

    if shared:  # at least one common .py file
        agent_text = "\n".join(agent_patch[f] for f in sorted(shared))
        gt_text    = "\n".join(gt_patch[f]    for f in sorted(shared))
        used_files = sorted(shared)
    else:       # no overlap → concatenate everything
        agent_text = "\n".join(agent_patch.values())
        gt_text    = "\n".join(gt_patch.values())
        used_files = []          # signal: full concatenation

    return agent_text, gt_text, used_files

###############################################################################
# Main program
###############################################################################
def main(folder: pathlib.Path) -> None:
    json_paths = sorted(folder.glob("*.json"))
    if not json_paths:
        print("No *.json files found in", folder)
        return

    distances: List[int] = []
    for jp in json_paths:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            agent_patch = data["agent_patch"]
            gt_patch    = data["ground_truth_patch"]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[SKIP] {jp.name}: cannot parse expected fields ({exc})")
            continue

        a_text, g_text, common = choose_patch_strings(agent_patch, gt_patch)
        dist = levenshtein(a_text, g_text)
        distances.append(dist)

        common_msg = ", ".join(common) if common else "<all files combined>"
        print(f"{jp.name:<40}  dist={dist:>6}  compared={common_msg}")

    if distances:
        avg = statistics.mean(distances)
        print(f"\nProcessed {len(distances)} file(s). "
              f"Average edit distance = {avg:.2f}")

###############################################################################
# Entry point
###############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute edit distance between agent and "
                    "ground-truth patches in JSON files.")
    parser.add_argument("folder", type=pathlib.Path,
                        help="Folder containing the result JSON files")
    args = parser.parse_args()
    main(args.folder.resolve())
