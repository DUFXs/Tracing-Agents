#!/usr/bin/env python3
import json
import argparse
import pandas as pd
import ast

def load_json(path):
    """Flatten JSON to DataFrame with issue_id, agent, resolved flag."""
    with open(path, 'r') as f:
        data = json.load(f)

    records = []
    for bucket in data.values():
        for issue_id, runs in bucket.items():
            for agent, details in runs.items():
                res = details.get('resolved')
                records.append({
                    'issue_id': issue_id,
                    'agent':    agent,
                    'resolved': bool(res)  # None -> False
                })
    return pd.DataFrame(records)


def load_parquet(path):
    """Read Parquet, parse FAIL_TO_PASS into list, and count failures."""
    df = pd.read_parquet(path)
    df = df.rename(columns={
        'instance_id': 'issue_id',
        'FAIL_TO_PASS': 'fail_list'
    })
    # parse stringified lists
    def parse_fail_list(x):
        if isinstance(x, str):
            try:
                lst = ast.literal_eval(x)
                return lst if isinstance(lst, list) else []
            except Exception:
                return []
        elif isinstance(x, list):
            return x
        else:
            return []
    df['fail_list'] = df['fail_list'].apply(parse_fail_list)
    df['fail_count'] = df['fail_list'].apply(len)
    return df[['issue_id', 'fail_count']]


def summarize(df_runs, df_fail, agent_name):
    """
    Produce summary: for each fail_count size:
      - total issues with that size
      - number resolved by given agent
    """
    # merge runs with failure counts
    df = df_fail.merge(df_runs[df_runs['agent'] == agent_name], on='issue_id', how='left')
    # unresolved runs will have NaN resolved, treat as False
    df['resolved'] = df['resolved'].fillna(False)

    # total issues per fail_count
    total = df_fail.groupby('fail_count')['issue_id'].nunique().rename('total_issues')
    # resolved per fail_count for this agent
    resolved = (
        df[df['agent'] == agent_name]
        .groupby('fail_count')['resolved']
        .sum()
        .rename('resolved_count')
    )
    # combine into one DataFrame
    summary = pd.concat([total, resolved], axis=1).fillna(0).astype(int)
    summary.index.name = 'fail_count'
    return summary.reset_index()


def main():
    p = argparse.ArgumentParser(
        description="Summarize resolution counts per FAIL_TO_PASS size for a specific agent."
    )
    p.add_argument('--parquet', required=True,
                   help="Parquet file path with instance_id and FAIL_TO_PASS.")
    p.add_argument('--json',    required=True,
                   help="JSON file path with nested results.")
    p.add_argument('--agent',   required=True,
                   help="Agent name to filter (e.g., '20241202_agentless-1.5_claude-3.5-sonnet').")
    args = p.parse_args()

    df_runs = load_json(args.json)
    df_fail = load_parquet(args.parquet)
    summary = summarize(df_runs, df_fail, args.agent)

    # print
    print(f"Summary for agent: {args.agent}\n")
    print("fail_count  total_issues  resolved_count")
    for _, row in summary.iterrows():
        print(f"{row['fail_count']:>10}  {row['total_issues']:>12}  {row['resolved_count']:>14}")

if __name__ == '__main__':
    main()
