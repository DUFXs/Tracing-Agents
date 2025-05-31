#!/usr/bin/env python3
import argparse
import json
import re
import ast
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def extract_issue_id(filename: str) -> str:
    match = re.match(r'^clone_detection_result_([^_]+__[^_]+)_', filename)
    return match.group(1) if match else None


def _clean(raw: str) -> str:
    return re.sub(r"[\\\"'\n\r\t]", "", raw).lower()


def extract_classification(response) -> str:
    if isinstance(response, dict):
        return _clean(str(response.get("classification", ""))) or "unknown"
    if not isinstance(response, str):
        return "unknown"
    classification = None
    try:
        inner = json.loads(response)
        classification = inner.get("classification")
    except json.JSONDecodeError:
        pass
    if classification is None:
        try:
            inner = ast.literal_eval(response)
            if isinstance(inner, dict):
                classification = inner.get("classification")
        except (ValueError, SyntaxError):
            pass
    if classification is None:
        m = re.search(r"[\"']?classification[\"']?\s*:\s*[\"']([^\"']+)[\"']", response,
                      flags=re.IGNORECASE|re.DOTALL)
        if m:
            classification = m.group(1)
    return _clean(str(classification)) if classification is not None else "unknown"


def load_difficulty_map(parquet_path: Path) -> dict:
    df = pd.read_parquet(parquet_path)
    if not {'instance_id', 'difficulty'}.issubset(df.columns):
        raise ValueError("Parquet must have 'instance_id' and 'difficulty' columns")
    return dict(zip(df['instance_id'], df['difficulty']))


def build_counts(folder: Path, difficulty_map: dict) -> tuple[dict, int]:
    counts = defaultdict(Counter)
    total = 0
    for fp in folder.glob('*.json'):
        issue_id = extract_issue_id(fp.name)
        if not issue_id:
            continue
        total += 1
        data = json.load(fp.open('r', encoding='utf-8'))
        diff = difficulty_map.get(issue_id)
        if diff is None or diff.strip().startswith('>'):
            continue
        clone_type = extract_classification(data.get('response', ''))
        counts[diff][clone_type] += 1
    return counts, total


def sort_difficulties(diffs):
    def key(d):
        low = d.replace(' ', '').lower()
        if low.startswith('<15min'): return 0
        if '15min' in low and '1hr' in low: return 1
        if '1hr' in low and '4hr' in low: return 2
        return 99
    return sorted(diffs, key=key)


def plot_by_difficulty(counts: dict, difficulties: list, outpath: Path) -> None:
    types = sorted({t for diff in counts for t in counts[diff].keys()}, key=lambda t: (t != 'unknown', t))
    cmap = plt.get_cmap('cividis')
    colors = [cmap(v) for v in np.linspace(0.1, 0.9, len(types))]

    # Use IEEE-like font (Times New Roman) and larger 30pt
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 30
    })

    fig, ax = plt.subplots(figsize=(14, 10), dpi=120)
    lefts = [0] * len(difficulties)

    # Remove left and right spines, keep top and bottom
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    def text_color(rgb):
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return 'black' if lum > 0.6 else 'white'

    for idx, t in enumerate(types):
        pct = []
        for diff in difficulties:
            total = sum(counts[diff].values())
            val = counts[diff].get(t, 0)
            pct.append((val / total * 100) if total > 0 else 0)
        ax.barh(difficulties, pct, left=lefts, color=colors[idx], label=t)
        for i, (l, p) in enumerate(zip(lefts, pct)):
            if p > 0:
                col = text_color(colors[idx])
                ax.text(l + p / 2, i, f"{p:.0f}%", va='center', ha='center', color=col)
        lefts = [l + p for l, p in zip(lefts, pct)]

    ax.set_xlabel('Percentage', fontsize=30)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis='x', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=30)

    # Bottom legend without frame
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.4, -0.15),
                       ncol=len(types), fontsize=30)
    legend.set_frame_on(False)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot horizontal percentage breakdown of clone-types by difficulty"
    )
    parser.add_argument('json_folder', type=Path, help='Folder with JSON results')
    parser.add_argument('parquet_file', type=Path, help='Parquet file path')
    parser.add_argument('--output', type=Path, default=Path('plots/clone_by_difficulty.png'), help='Output plot path')
    args = parser.parse_args()

    if not args.json_folder.is_dir():
        raise SystemExit(f"{args.json_folder} is not a directory")

    diff_map = load_difficulty_map(args.parquet_file)
    counts, total = build_counts(args.json_folder, diff_map)

    if total == 0:
        print("No JSON files processed. Check folder path and filename patterns.")
        return

    diffs = sort_difficulties(counts.keys())
    if not diffs:
        print("No difficulty categories found (after filtering).")
        return

    plot_by_difficulty(counts, diffs, args.output)

if __name__ == '__main__':
    main()
