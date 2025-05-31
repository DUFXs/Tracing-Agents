#!/usr/bin/env python3
import json
import argparse
import pandas as pd
import ast


def load_json(path):
    """Walk the nested JSON, count null as not resolved, and return a flat DataFrame of runs."""
    with open(path, 'r') as f:
        data = json.load(f)

    records = []
    for bucket in data.values():
        for issue_id, runs in bucket.items():
            for agent, details in runs.items():
                # Treat `null` as not resolved (False)
                res = details.get('resolved')
                records.append({
                    'issue_id': issue_id,
                    'agent':    agent,
                    'resolved': int(bool(res)),  # None -> False -> 0
                })
    return pd.DataFrame.from_records(records)


def load_parquet(path):
    """Read the Parquet, rename instance_id→issue_id, parse FAIL_TO_PASS, and count failures."""
    df = pd.read_parquet(path)
    df = df.rename(columns={
        'instance_id': 'issue_id',
        'FAIL_TO_PASS': 'fail_list'
    })
    # If fail_list is stored as a string representation of a list, parse it
    def parse_fail_list(x):
        if isinstance(x, str):
            try:
                parsed = ast.literal_eval(x)
                return parsed if isinstance(parsed, list) else []
            except (ValueError, SyntaxError):
                return []
        elif isinstance(x, list):
            return x
        else:
            return []
    df['fail_list'] = df['fail_list'].apply(parse_fail_list)
    df['fail_count'] = df['fail_list'].apply(len)
    return df[['issue_id', 'fail_count']]


def summarize_and_corr(df_runs, df_fail):
    df = df_runs.merge(df_fail, on='issue_id', how='left')
    out = []
    for agent, grp in df.groupby('agent'):
        n_runs       = len(grp)
        unresolved   = (grp['resolved'] == 0).sum()
        resolved_cnt = (grp['resolved'] == 1).sum()
        std_fail_ct  = grp['fail_count'].std()
        std_resolved = grp['resolved'].std()
        pearson_r    = grp['resolved'].corr(grp['fail_count'])
        total_fail_ct= grp['fail_count'].sum()
        out.append({
            'agent':           agent,
            'n_runs':          n_runs,
            '#unresolved':     unresolved,
            '#resolved':       resolved_cnt,
            'total_fail_ct':   total_fail_ct,
            'std_fail_ct':     std_fail_ct,
            'std_resolved':    std_resolved,
            'pearson_r':       pearson_r
        })
    return pd.DataFrame(out).sort_values('agent')


def main():
    p = argparse.ArgumentParser(
        description="Compute correlation between resolve-rate and #failures per agent"
    )
    p.add_argument('--parquet', required=True,
                   help="Path to your issues.parquet (with instance_id & FAIL_TO_PASS columns)")
    p.add_argument('--json',    required=True,
                   help="Path to your nested results.json")
    args = p.parse_args()

    df_runs = load_json(args.json)
    df_fail = load_parquet(args.parquet)
    summary = summarize_and_corr(df_runs, df_fail)

    # Print a nice table including total failures
    print(summary.to_string(index=False, float_format="%.3f"))

if __name__ == '__main__':
    main()
