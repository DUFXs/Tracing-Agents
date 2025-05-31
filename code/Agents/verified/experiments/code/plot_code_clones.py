#!/usr/bin/env python3
"""
plot_clone_distribution.py

Create a bar plot showing the distribution of clone–type classifications
(Type‑1, Type‑2, …) produced by a given agent.

USAGE
-----
python plot_clone_distribution.py /path/to/json_folder
"""
from __future__ import annotations

import argparse
import json
import re
import ast
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# ──────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────

# def extract_classification(response_text: str) -> str:
#     """Parse the nested JSON in the *response* field and return the cleaned
#     *classification* value (e.g., "type-2").  If parsing fails, return
#     "unknown".
#     """
#     try:
#         inner = json.loads(response_text)
#         raw = inner.get("classification", "").strip()
#         # Remove stray escape sequences or quotes
#         return re.sub(r"[\\\"'\n\r\t]", "", raw).lower()
#     except json.JSONDecodeError:
#         print(response_text)
#         return "unknown"

def _clean(raw: str) -> str:
    """Strip escape chars / quotes and lowercase."""
    return re.sub(r"[\\\"'\n\r\t]", "", raw).lower()


def extract_classification(response: Any) -> str:
    """Return the *classification* label from *response*.

    *response* may be a dict or a JSON / Python‑dict‑literal *string*.
    If every structured parse fails we search with regex as a last resort.
    """
    # -------------------------------------------
    # If already a mapping
    # -------------------------------------------
    if isinstance(response, dict):
        return _clean(str(response.get("classification", ""))) or "unknown"

    # -------------------------------------------
    # If it's not a string we can't do much
    # -------------------------------------------
    if not isinstance(response, str):
        return "unknown"

    classification: str | None = None

    # -------------------------------------------
    # 1) Strict JSON
    # -------------------------------------------
    try:
        inner = json.loads(response)
        classification = inner.get("classification")
    except json.JSONDecodeError:
        pass

    # -------------------------------------------
    # 2) Lenient Python literal eval
    # -------------------------------------------
    if classification is None:
        try:
            inner = ast.literal_eval(response)
            if isinstance(inner, dict):
                classification = inner.get("classification")
        except (ValueError, SyntaxError):
            pass

    # -------------------------------------------
    # 3) Regex fallback inside *response* string
    # -------------------------------------------
    if classification is None:
        cls_match = re.search(
            r"[\"']?classification[\"']?\s*:\s*[\"']([^\"']+)[\"']",
            response,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if cls_match:
            classification = cls_match.group(1)

    # Final cleaning & default
    if classification is None:
        return "unknown"

    cleaned = _clean(str(classification))
    return cleaned or "unknown"


# ──────────────────────────────────────────────────────────────────────────
# Core logic
# ──────────────────────────────────────────────────────────────────────────

def build_distribution(folder: Path) -> tuple[Counter, str]:
    """Walk through *folder*, reading every *.json file and accumulating a
    Counter of clone‑type classifications.  Returns the Counter and the
    *agent* string (taken from the first file encountered).
    """
    counts: Counter = Counter()
    agent_name: str | None = None

    for fp in folder.glob("*.json"):
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Capture the agent name once
        if agent_name is None:
            agent_name = data.get("agent", "unknown-agent")

        classification = extract_classification(data.get("response", ""))
        if classification:
            counts[classification] += 1

    if agent_name is None:
        agent_name = "unknown-agent"

    return counts, agent_name


def plot_distribution(counts: Counter, agent: str, outpath: Path) -> None:
    """Generate and save a bar chart visualising *counts*.

    The plot follows a clean publication style: high contrast, minimal
    chartjunk, exact value labels, readable fonts.
    """
    # Consistent ordering: type-1, type-2, …, unknown
    ordered = sorted(counts.items(), key=lambda kv: (kv[0] != "unknown", kv[0]))
    labels, values = zip(*ordered)
    total = sum(values)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)

    # Draw bars
    bars = ax.bar(
        range(len(labels)),
        values,
        width=0.6,
        edgecolor="black",
        linewidth=0.8,
    )

    # Annotate each bar with count and percentage
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val}  ({val / total:.0%})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Axes configuration
    ax.set_xticks(range(len(labels)), [lbl.capitalize() for lbl in labels], fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Distribution of code clones for Agentless on Claude-3.5-Sonnet", fontsize=14, pad=12)
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"✓ Saved figure → {outpath}")


# ──────────────────────────────────────────────────────────────────────────
# Entry‑point
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot clone-type distribution for a given agent's results folder.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing result JSON files",
    )
    args = parser.parse_args()

    if not args.folder.is_dir():
        raise SystemExit(f"❌ Path “{args.folder}” is not a directory.")

    counts, agent = build_distribution(args.folder)
    if not counts:
        raise SystemExit("❌ No classification data found.")

    # NEW ➜ ensure a “plots” sub‑folder and save figure there
    out_dir = args.folder / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    outpng = out_dir / f"clone_distribution_{agent}.png"
    plot_distribution(counts, agent, outpng)


if __name__ == "__main__":
    main()
