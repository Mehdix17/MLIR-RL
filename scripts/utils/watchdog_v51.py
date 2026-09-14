#!/usr/bin/env python3
"""Watchdog for the V5.1 runs (ops_and_blocks + legacy_paper, transformer + LSTM).
Prints ONLY on anomalies.

Checks: at least one driver RUNNING, dask worker fleet healthy, bench failures in the
latest log, cross-worker duplicate execs, and progress.
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime

REPO = '/scratch/mb10856/MLIR-RL'
os.chdir(REPO)
alerts = []

try:
    out = subprocess.run(['squeue', '-u', os.environ.get('USER', 'mb10856'), '-h', '-o', '%i %j %t'],
                         capture_output=True, text=True, timeout=30).stdout
except Exception as e:
    print(f'[{datetime.now():%m-%d %H:%M}] squeue failed: {e}')
    sys.exit(1)

jobs = {}
for line in out.strip().splitlines():
    parts = line.split()
    if len(parts) == 3:
        jobs[parts[0]] = (parts[1], parts[2])
dask_workers = [j for j, (n, s) in jobs.items() if n == 'dask' and s == 'R']
drivers_running = [j for j, (n, s) in jobs.items() if n == 'mlir-train' and s == 'R']

if len(drivers_running) < 2:
    alerts.append(f'DRIVERS: {len(drivers_running)} mlir-train jobs running (<2)')
if len(dask_workers) < 16:
    alerts.append(f'WORKERS: {len(dask_workers)} running (< 16)')

# bench failures / duplicate execs from the latest distributed log
dist_log = max([f for f in os.listdir(f'{REPO}/logs') if f.startswith('train_')],
               key=lambda f: os.path.getmtime(f'{REPO}/logs/{f}'), default=None)
if dist_log:
    tail = subprocess.run(['tail', '-200', f'{REPO}/logs/{dist_log}'],
                          capture_output=True, text=True).stdout
    fails = re.findall(r'(\d+) benchmark failures this iteration', tail)
    dups = re.findall(r'(\d+) cross-worker duplicate execs', tail)
    if fails and int(fails[-1]) > 5:
        alerts.append(f'bench failures: {fails[-1]}/iter (>5)')
    if dups and int(dups[-1]) > 10:
        alerts.append(f'duplicate execs: {dups[-1]}/iter')

if alerts:
    print(f'[{datetime.now():%m-%d %H:%M}] V5.1 WATCHDOG: ' + ' | '.join(alerts))
