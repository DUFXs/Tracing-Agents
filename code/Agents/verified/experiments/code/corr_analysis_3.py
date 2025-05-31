#!/usr/bin/env python3
import json
import argparse
import pandas as pd
import ast


def load_json(path):
    """Flatten the nested JSON into a DataFrame of (instance_id, agent, resolved_int)."""
    with open(path, 'r') as f:
        data = json.load(f)

    records = []
    for bucket in data.values():
        for instance_id, runs in bucket.items():
            for agent_name, details in runs.items():
                resolved = details.get('resolved')
                if resolved is not None:
                    records.append({
                        'instance_id':  instance_id,
                        'agent':        agent_name,
                        'resolved_int': int(bool(resolved))
                    })
    return pd.DataFrame(records)


def load_parquet(path):
    """Read the Parquet, parse FAIL_TO_PASS column, extract difficulty, return instance_id, test count, and difficulty."""
    df = pd.read_parquet(path)
    # Rename for clarity
    df = df.rename(columns={'FAIL_TO_PASS': 'fail_list'})

    def parse_fail_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                try:
                    return json.loads(val)
                except Exception:
                    return []
        return []

    # Parse and count failed-to-pass tests
    df['parsed_fail_list'] = df['fail_list'].apply(parse_fail_list)
    df['num_FAIL_TO_PASS_tests'] = df['parsed_fail_list'].apply(len)

    # Ensure difficulty column exists in Parquet
    if 'difficulty' not in df.columns:
        raise KeyError("Column 'difficulty' not found in the parquet file")

    return df[['instance_id', 'num_FAIL_TO_PASS_tests', 'difficulty']]


def pivot_wide(df_runs, df_fail):
    """
    Merge runs + failure count + difficulty, then pivot so each agent is its own column of resolved_int.
    Unseen instance/agent combos become NaN (or fill with 0).
    """
    merged = df_runs.merge(df_fail, on='instance_id', how='left')
    wide = merged.pivot_table(
        index=['instance_id', 'num_FAIL_TO_PASS_tests', 'difficulty'],
        columns='agent',
        values='resolved_int',
        aggfunc='first'
    )
    # Flatten the column index and reset
    wide.columns.name = None
    wide = wide.reset_index()
    return wide


def main():
    p = argparse.ArgumentParser(
        description="Build a wide-format table: one row per instance, num_FAIL_TO_PASS_tests, difficulty, + resolved per agent"
    )
    p.add_argument('--parquet', required=True, help="Path to your data.parquet file")
    p.add_argument('--json',    required=True, help="Path to your nested results.json file")
    p.add_argument('--out',     default='resolution_by_fail_to_pass.csv',
                   help="Output CSV filename")
    args = p.parse_args()

    df_runs = load_json(args.json)
    df_fail = load_parquet(args.parquet)
    df_wide = pivot_wide(df_runs, df_fail)

    # Fill missing resolved flags with 0 and cast to small integer
    df_wide = df_wide.fillna(0).astype(
        {c: 'int8' for c in df_wide.columns if c not in ['instance_id', 'num_FAIL_TO_PASS_tests', 'difficulty']}
    )

    print(f"Writing wide-format table with {len(df_wide)} instances → {args.out}")
    df_wide.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
