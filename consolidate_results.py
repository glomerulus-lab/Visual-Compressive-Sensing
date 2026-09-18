#!/usr/bin/env python
"""Concatenate per-run sweep CSVs into one consolidated CSV per observation.

hyperparam_sweep_filter.py writes one timestamped CSV per invocation:

    result/<algorithm>/<method>/<image>/<observation>/{True,False}_param_<ctime>.csv

This collects those into the consolidated file the analysis code reads:

    result/<algorithm>/<method>/<image>/<observation>/{V1,Pixel,Gaussian}.csv

The True_/False_ prefix on the inputs is `color`, not `fixed_weights`; runs of
both kinds are concatenated together, exactly as the existing files are.

SAFETY
------
The consolidated files already in this repo are NOT reproducible from the
timestamped CSVs sitting beside them -- checked on
result/lasso/dct/baboon/gaussian,
where 0 of the 600 consolidated rows appear anywhere in the 1200 timestamped
rows, and the consolidated file carries filter_dim values ((8,8), (16,16)) that
no surviving per-run file contains.  The per-run CSVs for those sweeps are
gone.  So a blind rebuild would DESTROY data with no other source.

Accordingly this script never overwrites an existing consolidated file unless
told to, and always writes a timestamped .bak first:

    (default)   build only where no consolidated file exists yet
    --merge     union existing + per-run rows, drop exact duplicates, rewrite
    --rebuild   rewrite from per-run CSVs alone  (LOSSY -- see above)
    --dry-run   report only, touch nothing

Usage:
    python consolidate_results.py --dry-run
    python consolidate_results.py --merge
    python consolidate_results.py --method dct --image barbara --merge
    python consolidate_results.py --algorithm bp --merge
"""

import argparse
import glob
import os
import shutil
import time

import pandas as pd

# directory name -> consolidated filename stem
OBS_FILENAME = {'V1': 'V1', 'pixel': 'Pixel', 'gaussian': 'Gaussian'}


def per_run_csvs(obs_dir):
    """The timestamped per-run CSVs in one observation directory."""
    return sorted(glob.glob(os.path.join(obs_dir, '*_param_*.csv')))


def consolidated_path(obs_dir):
    """Where the consolidated CSV for this observation directory lives."""
    obs = os.path.basename(obs_dir)
    stem = OBS_FILENAME.get(obs, obs.capitalize())
    return os.path.join(obs_dir, f'{stem}.csv')


def param_columns(df):
    return [c for c in df.columns if c != 'error']


def load_and_concat(paths):
    """Concatenate CSVs that may disagree on columns.

    Older per-run files predate the filter_dim parameter and lack that column;
    pandas fills it with NaN rather than dropping the rows.  Those gaps are
    reported rather than guessed at -- filter_dim tracked the num_cell wave
    (8-32 -> (8,8), 32-128 -> (16,16), 128-512 -> (32,32)), but that is an
    inference from the data, not something the files record.
    """
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else None


def tidy(df):
    """Drop exact duplicate rows and sort by the parameter columns."""
    df = df.drop_duplicates(ignore_index=True)
    par = [c for c in param_columns(df) if c in df.columns]
    if par:
        df = df.sort_values(par, kind='stable', na_position='last')
    return df.reset_index(drop=True)


def backup(path):
    stamp = time.strftime('%Y%m%d_%H%M%S')
    dest = f'{path}.bak.{stamp}'
    shutil.copy2(path, dest)
    return dest


def process(obs_dir, mode, dry_run, root):
    runs = per_run_csvs(obs_dir)
    out = consolidated_path(obs_dir)
    exists = os.path.exists(out)
    label = os.path.relpath(obs_dir, root)

    if not runs and not exists:
        return None
    if not runs:
        print(f"  {label:<26} no per-run CSVs; leaving {os.path.basename(out)} alone")
        return None

    new = load_and_concat(runs)

    if exists and mode == 'skip':
        print(f"  {label:<26} SKIP  {os.path.basename(out)} exists "
              f"({sum(1 for _ in open(out)) - 1} rows); use --merge or --rebuild")
        return None

    if exists and mode == 'merge':
        combined = tidy(pd.concat([pd.read_csv(out), new], ignore_index=True, sort=False))
        before = sum(1 for _ in open(out)) - 1
    else:                                   # rebuild, or nothing there yet
        combined = tidy(new)
        before = (sum(1 for _ in open(out)) - 1) if exists else 0

    missing = int(combined['filter_dim'].isna().sum()) if 'filter_dim' in combined else 0
    note = f"  [{missing} rows missing filter_dim]" if missing else ""
    action = 'would write' if dry_run else 'wrote'
    print(f"  {label:<26} {len(runs)} run files -> {len(combined):>5} rows "
          f"(was {before}){note}")

    if dry_run:
        return None
    if exists:
        print(f"      backup: {os.path.basename(backup(out))}")
    combined.to_csv(out, index=False)
    print(f"      {action}: {os.path.basename(out)}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument('--merge', action='store_true',
                   help='union existing consolidated rows with per-run rows')
    g.add_argument('--rebuild', action='store_true',
                   help='rewrite from per-run CSVs alone (LOSSY, see docstring)')
    ap.add_argument('--dry-run', action='store_true', help='report only')
    ap.add_argument('--algorithm',
                    help='limit to one solver subtree (lasso, bp, ...)')
    ap.add_argument('--method', help='limit to one method (dct, dwt)')
    ap.add_argument('--image', help='limit to one image')
    ap.add_argument('--result-dir', default='result')
    args = ap.parse_args()

    mode = 'merge' if args.merge else 'rebuild' if args.rebuild else 'skip'
    pattern = os.path.join(args.result_dir, args.algorithm or '*',
                           args.method or '*', args.image or '*', '*')

    print(f"mode={mode}{' (dry run)' if args.dry_run else ''}  scanning {pattern}\n")
    n = 0
    for obs_dir in sorted(glob.glob(pattern)):
        if os.path.isdir(obs_dir):
            n += process(obs_dir, mode, args.dry_run, args.result_dir) is not None
    print(f"\n{n} consolidated file(s) written"
          if not args.dry_run else "\ndry run: nothing written")


if __name__ == '__main__':
    main()
