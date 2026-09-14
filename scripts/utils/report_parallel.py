#!/usr/bin/env python3
"""V5.1 paired-run comparison: single-node vs distributed PPO.

Parses the sbatch .out logs of both runs and prints a comparison table.

Usage:
  python scripts/utils/report_parallel.py --single logs/train_<id>.out --distributed logs/train_<id>.out
  python scripts/utils/report_parallel.py --single <out> --distributed <out> --single-job 1723... --dist-job 1723...
"""

import argparse
import re
import subprocess
import statistics


def _to_secs(s):
    parts = s.split(':')
    secs = float(parts[-1])
    if len(parts) > 1:
        secs += 60 * float(parts[-2])
    if len(parts) > 2:
        secs += 3600 * float(parts[-3])
    return secs


def parse_log(path):
    """Extract per-iteration wall time, collection time, PPO fit time from a train .out log."""
    iter_walls, collections, fits, transitions = [], [], [], []
    for line in open(path, errors='replace'):
        m = re.search(r"Main Loop \d+/\d+ .*\(([\d:\.]+)/it\)", line)
        if m:
            wall = _to_secs(m.group(1))
            if wall > 0:  # skip the pre-loop "(0/it)" placeholder line
                iter_walls.append(wall)
        m = re.search(r"(Distributed collection|Collecting 64 benchmarks):? ([0-9:\.]+).*?(\d+) transitions", line)
        if m:
            collections.append(_to_secs(m.group(2)))
            transitions.append(int(m.group(3)))
        m = re.search(r"PPO fit in ([0-9:\.]+)", line)
        if m:
            fits.append(_to_secs(m.group(1)))
    return iter_walls, collections, fits, transitions


def max_rss(job_id):
    if not job_id:
        return None
    out = subprocess.run(['sacct', '-j', job_id, '--format=MaxRSS', '-P', '-n'],
                         capture_output=True, text=True).stdout.strip().split('\n')
    return out[0] if out else None


def med(xs):
    return statistics.median(xs) if xs else float('nan')


def fmt(secs):
    return f"{secs:.1f}s" if secs == secs else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single', required=True, help='single-node train .out log')
    ap.add_argument('--distributed', required=True, help='distributed train .out log')
    ap.add_argument('--single-job', default='', help='single-node slurm job id (for MaxRSS)')
    ap.add_argument('--dist-job', default='', help='distributed driver slurm job id (for MaxRSS)')
    args = ap.parse_args()

    rows = {}
    for label, path, job in (('single-node', args.single, args.single_job),
                             ('distributed', args.distributed, args.dist_job)):
        walls, colls, fits, trans = parse_log(path)
        rows[label] = dict(iterations=len(walls), iter_med=med(walls),
                           collection_med=med(colls), fit_med=med(fits),
                           transitions_med=med(trans), maxrss=max_rss(job))

    print(f"{'metric':<22}{'single-node':>14}{'distributed':>14}")
    for key, label in (('iterations', 'iterations'), ('iter_med', 'iter wall (median)'),
                       ('collection_med', 'collection (median)'), ('fit_med', 'PPO fit (median)'),
                       ('transitions_med', 'transitions/iter'), ('maxrss', 'MaxRSS')):
        s, d = rows['single-node'][key], rows['distributed'][key]
        if key == 'transitions_med':
            sv, dv = f"{s:.0f}" if s == s else '-', f"{d:.0f}" if d == d else '-'
        elif key.endswith('_med'):
            sv, dv = fmt(s), fmt(d)
        else:
            sv, dv = str(s), str(d)
        print(f"{label:<22}{sv:>14}{dv:>14}")
    if rows['single-node']['iterations'] and rows['distributed']['iterations']:
        speedup = rows['single-node']['iter_med'] / rows['distributed']['iter_med']
        print(f"\nspeedup (single/distributed, median iter): {speedup:.2f}x")


if __name__ == '__main__':
    main()
