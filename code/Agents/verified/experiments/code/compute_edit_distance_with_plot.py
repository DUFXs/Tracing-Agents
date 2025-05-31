# #!/usr/bin/env python3
# """
# edit_distance_with_plot.py

# (1) Compute the Levenshtein edit distance between agent-patch and ground-truth
#     patch for every JSON file in a folder.
# (2) Join those distances with a Parquet file that provides a 'difficulty'
#     bucket per instance_id.
# (3) Generate a box-and-whisker figure of edit-distance by difficulty bucket.

# Usage
# -----
#     python edit_distance_with_plot.py <json_folder> <difficulty.parquet>
# """

# import argparse
# import json
# import pathlib
# from typing import Dict, List, Tuple

# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# ###############################################################################
# # Levenshtein (no external libs)
# ###############################################################################
# def levenshtein(a: str, b: str) -> int:
#     if a == b:
#         return 0
#     if len(a) < len(b):
#         a, b = b, a
#     previous = list(range(len(b) + 1))
#     for i, ca in enumerate(a, 1):
#         current = [i]
#         for j, cb in enumerate(b, 1):
#             current.append(
#                 min(previous[j] + 1,           # insertion
#                     current[j - 1] + 1,        # deletion
#                     previous[j - 1] + (ca != cb))  # substitute
#             )
#         previous = current
#     return previous[-1]

# ###############################################################################
# # Patch-selection logic (prompt spec)
# ###############################################################################
# def choose_patch_strings(agent_patch: Dict[str, str],
#                          gt_patch: Dict[str, str]) -> Tuple[str, str]:
#     agent_files = set(agent_patch)
#     shared = agent_files & set(gt_patch)
#     if shared:
#         a = "\n".join(agent_patch[f] for f in sorted(shared))
#         g = "\n".join(gt_patch[f]    for f in sorted(shared))
#     else:
#         a = "\n".join(agent_patch.values())
#         g = "\n".join(gt_patch.values())
#     return a, g

# ###############################################################################
# def collect_edit_distances(json_folder: pathlib.Path) -> pd.DataFrame:
#     """Return a DataFrame with columns: instance_id, edit_distance."""
#     rows = []
#     for jf in sorted(json_folder.glob("*.json")):
#         try:
#             d = json.loads(jf.read_text(encoding="utf-8"))
#             a_txt, g_txt = choose_patch_strings(d["agent_patch"],
#                                                 d["ground_truth_patch"])
#             dist = levenshtein(a_txt, g_txt)
#             rows.append({"instance_id": d["instance_id"],
#                          "edit_distance": dist})
#         except Exception as exc:
#             print(f"[SKIP] {jf.name}: {exc}")
#     return pd.DataFrame(rows)

# ###############################################################################
# def make_boxplot(df: pd.DataFrame, outfile: pathlib.Path) -> None:
#     """Create a box-and-whisker plot of edit_distance by difficulty bucket."""
#     buckets = list(df["difficulty"].dropna().unique())
#     buckets.sort(key=lambda b: df.loc[df["difficulty"] == b,
#                                       "edit_distance"].median())

#     # Build list-of-arrays for matplotlib
#     data = [df.loc[df["difficulty"] == b, "edit_distance"].values
#             for b in buckets]

#     plt.figure(figsize=(6, 4))
#     plt.boxplot(data, positions=np.arange(len(buckets))+1,
#                 widths=0.55, patch_artist=False, showfliers=False)

#     # Raw points (jitter)
#     for i, vals in enumerate(data, start=1):
#         jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
#         plt.scatter(np.full_like(vals, i) + jitter, vals,
#                     alpha=0.6, marker="x")

#     plt.xticks(np.arange(len(buckets))+1, buckets)
#     plt.xlabel("Time-to-fix bucket")
#     plt.ylabel("Edit distance (characters)")
#     plt.title("Edit-distance distribution by difficulty")
#     plt.tight_layout()
#     plt.savefig(outfile, dpi=300)
#     plt.close()
#     print(f"Figure written to {outfile}")

# ###############################################################################
# def main(json_folder: pathlib.Path, parquet_path: pathlib.Path) -> None:
#     distances_df = collect_edit_distances(json_folder)
#     if distances_df.empty:
#         print("No valid JSON files processed.")
#         return

#     diff_df = pd.read_parquet(parquet_path,
#                               columns=["instance_id", "difficulty"])
#     merged = distances_df.merge(diff_df, on="instance_id", how="left")

#     # Print per-bucket summary
#     summary = (merged.groupby("difficulty")["edit_distance"]
#                       .agg(["count", "median", "mean"])
#                       .sort_index())
#     print("\nEdit-distance summary by difficulty bucket:")
#     print(summary.to_string(float_format="%.1f"))

#     # Plot
#     make_boxplot(merged, json_folder / "edit_distance_boxplot.png")

# ###############################################################################
# if __name__ == "__main__":
#     argp = argparse.ArgumentParser(description=__doc__)
#     argp.add_argument("json_folder", type=pathlib.Path,
#                       help="Folder containing the result JSON files")
#     argp.add_argument("difficulty_parquet", type=pathlib.Path,
#                       help="Parquet file with columns [instance_id, difficulty]")
#     args = argp.parse_args()
#     main(args.json_folder.resolve(), args.difficulty_parquet.resolve())

#!/usr/bin/env python3
"""
edit_distance_with_plot.py  –  v3.1  (log-axis fig, Times New Roman 30 pt, dark ‘×’)

• Computes Levenshtein edit distance for every JSON file in <json_folder>.
• Joins those distances with <difficulty.parquet> (instance_id, difficulty).
• Writes:
    ├─ edit_distance_summary.csv
    └─ edit_distance_boxplot.png   (log-scaled, 30 pt Times New Roman, dark points)

Usage
-----
    python edit_distance_with_plot.py  <json_folder>  <difficulty.parquet>
"""

import argparse
import json
import pathlib
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# GLOBAL STYLING – Times New Roman, 30 pt
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "Times New Roman",
    "font.size":       30,
    "axes.titlesize":  30,
    "axes.labelsize":  30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
})

# ──────────────────────────────────────────────────────────────
# Levenshtein (no external libs)
# ──────────────────────────────────────────────────────────────
def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(prev[j] + 1,           # insert
                    curr[j - 1] + 1,       # delete
                    prev[j - 1] + (ca != cb))  # substitute
            )
        prev = curr
    return prev[-1]

# ──────────────────────────────────────────────────────────────
def choose_patch_strings(agent_patch: Dict[str, str],
                         gt_patch: Dict[str, str]) -> Tuple[str, str]:
    shared = set(agent_patch) & set(gt_patch)
    if shared:
        a_txt = "\n".join(agent_patch[f] for f in sorted(shared))
        g_txt = "\n".join(gt_patch[f]    for f in sorted(shared))
    else:
        a_txt = "\n".join(agent_patch.values())
        g_txt = "\n".join(gt_patch.values())
    return a_txt, g_txt

# ──────────────────────────────────────────────────────────────
def collect_distances(json_folder: pathlib.Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for jf in sorted(json_folder.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
            a_txt, g_txt = choose_patch_strings(d["agent_patch"],
                                                d["ground_truth_patch"])
            rows.append({
                "instance_id":  d["instance_id"],
                "edit_distance": levenshtein(a_txt, g_txt)
            })
        except Exception as exc:
            print(f"[SKIP] {jf.name}: {exc}")
    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────
def make_log_boxplot(df: pd.DataFrame, outfile: pathlib.Path) -> None:
    """Box-and-whisker on log axis; jittered points & bucket counts."""
    buckets = sorted(df["difficulty"].dropna().unique())
    data = [df.loc[df["difficulty"] == b, "edit_distance"].values
            for b in buckets]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(data,
               positions=np.arange(len(buckets)) + 1,
               widths=0.55, patch_artist=False,
               showfliers=False)

    # Dark ‘×’ marks
    rng = np.random.default_rng(0)
    for i, vals in enumerate(data, start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full_like(vals, i) + jitter,
                   vals,
                   alpha=0.85,
                   color="black",
                   marker="x",
                   linewidths=1.0)

    # Log-scale y-axis
    if (df["edit_distance"] == 0).any():
        df.loc[df["edit_distance"] == 0, "edit_distance"] = 0.5
    ax.set_yscale("log")
    ax.set_ylabel("Edit distance (chars, log scale)")

    # X-tick labels (bucket + count)
    xticklabels = [f"{b}\n(n={len(v)})" for b, v in zip(buckets, data)]
    ax.set_xticks(np.arange(len(buckets)) + 1)
    ax.set_xticklabels(xticklabels, ha="center")

    #ax.set_title("Edit-distance distribution by difficulty (log axis)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Figure saved → {outfile}")

# ──────────────────────────────────────────────────────────────
def main(json_folder: pathlib.Path, parquet_file: pathlib.Path) -> None:
    dist_df = collect_distances(json_folder)
    if dist_df.empty:
        print("No JSON files processed.")
        return

    diff_df = pd.read_parquet(parquet_file,
                              columns=["instance_id", "difficulty"])
    merged = dist_df.merge(diff_df, on="instance_id", how="left")

    # Bucket summary CSV
    summary = (merged.groupby("difficulty")["edit_distance"]
                      .agg(count="size", median="median", mean="mean"))
    summary_path = json_folder / "edit_distance_summary.csv"
    summary.to_csv(summary_path, index=True)
    print(f"Summary saved → {summary_path}")

    # Plot
    make_log_boxplot(merged, json_folder / "edit_distance_boxplot.png")

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute edit distances and plot by difficulty bucket "
                    "(Times New Roman, 30 pt, dark markers)."
    )
    parser.add_argument("json_folder", type=pathlib.Path,
                        help="Folder containing JSON result files")
    parser.add_argument("difficulty_parquet", type=pathlib.Path,
                        help="Parquet file mapping instance_id → difficulty")
    args = parser.parse_args()
    main(args.json_folder.resolve(), args.difficulty_parquet.resolve())
