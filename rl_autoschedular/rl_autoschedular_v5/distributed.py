"""Distributed PPO trajectory collection (driver + Dask workers).

The driver keeps the model and runs the PPO update; workers step their group of
benchmarks with a local replica of the current weights and return trajectory
segments. One shared brain, updated once per iteration on the full trajectory.

Enabled when DASK_NODES > 0 (see scripts/train/train.py); the single-node path
(ppo.collect_trajectory) is untouched.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from time import time
from typing import Optional

import torch

from rl_autoschedular_v5 import device
from rl_autoschedular_v5.benchmarks import Benchmarks
from rl_autoschedular_v5.env import Env
from rl_autoschedular_v5.execution import Execution
from rl_autoschedular_v5.model import HiearchyModel
from rl_autoschedular_v5.ppo import __execute_states, rollout_benchmarks
from rl_autoschedular_v5.trajectory import TrajectoryCollector
from rl_autoschedular_v5.utils.config import Config
from rl_autoschedular_v5.utils.dask_manager import DaskManager
from rl_autoschedular_v5.utils.file_logger import FileLogger
from rl_autoschedular_v5.utils.log import print_alert, print_error, print_info

# NOTE: no SIGABRT handler here — this module is imported inside Dask worker
# threads where signal.signal() is illegal, and a dead worker is recovered by
# the Dask nanny (group skipped) anyway. The driver installs its own handler.

# One group = 4 rollouts in parallel, each capped by MIN_EXEC_TIMEOUT (default 300s).
GROUP_TIMEOUT = int(os.getenv('DASK_GROUP_TIMEOUT', '600'))

_worker_data: Optional[tuple[Benchmarks, Optional[dict[str, dict[str, int]]]]] = None


def _get_worker_data() -> tuple[Benchmarks, Optional[dict[str, dict[str, int]]]]:
    """Benchmarks + main_exec_data, loaded once per worker process (Dask persists module state)."""
    global _worker_data
    if _worker_data is None:
        cfg = Config()
        main_exec_data = None
        if cfg.main_exec_data_file:
            with open(cfg.main_exec_data_file) as f:
                main_exec_data = json.load(f)
        _worker_data = (Benchmarks(), main_exec_data)
    return _worker_data


def rollout_group(
    group_idx: int,
    indices: list[int],
    weights: dict,
    step: int,
    exec_data_file: str,
) -> tuple[int, tuple[TrajectoryCollector, dict[str, dict[str, int]], list[float], list[float]]]:
    """Worker-side task: rollout + execute one group of benchmarks (4-parallel within node).

    Returns (group_idx, (TrajectoryCollector, new_cache_data, entropies, speedups)).
    """
    data, main_exec_data = _get_worker_data()

    model = HiearchyModel().to(device)
    model.load_state_dict(weights)

    envs, states, tcs, entropies = rollout_benchmarks(data, model, indices, step)

    # Execute the group's terminal states in parallel within this worker.
    with ThreadPoolExecutor(max_workers=len(states)) as pool:
        results = list(pool.map(
            lambda s: __execute_states(s, exec_data_file, data, main_exec_data), states
        ))

    results = [
        (*env.failed_seq(state.transformation_history), float(GROUP_TIMEOUT))
        if not r else r
        for r, env, state in zip(results, envs, states)
    ]
    all_rewards, all_speedups, all_exec_times, _, _ = tuple(zip(*results))

    new_cache_data: dict[str, dict[str, int]] = {}
    speedups: list[float] = []
    exe = Execution()
    for tc, state, rewards, speedup, exec_time in zip(tcs, states, all_rewards, all_speedups, all_exec_times):
        tc.rewards = rewards
        speedups.append(speedup)
        if exec_time is not None:
            cache_key = exe.get_code_cache_key(state.transformation_history)
            new_cache_data.setdefault(state.bench_name, {})[cache_key] = exec_time

    return group_idx, (sum(tcs, TrajectoryCollector()), new_cache_data, entropies, speedups)


def collect_distributed_trajectory(data: Benchmarks, model: HiearchyModel, step: int):
    """Distributed replacement for ppo.collect_trajectory (used when DASK_NODES > 0).

    Samples bench_count benchmarks, splits them into groups of 4, dispatches one
    group per worker node (round-robin; queued one-at-a-time per worker via
    single_task_slot), and returns the full trajectory. Failed groups are
    skipped — their benchmarks are absent from that iteration's trajectory.
    """
    dm = DaskManager()
    fl = FileLogger()
    exe = Execution()
    cfg = Config()

    indices = torch.randperm(len(data))[:cfg.bench_count].long().tolist()
    if len(indices) < cfg.bench_count:
        indices = (indices * cfg.bench_count)[:cfg.bench_count]
    groups = [indices[i:i + 4] for i in range(0, len(indices), 4)]

    workers = dm.workers_names
    if not workers:
        raise RuntimeError("Dask cluster has no workers — check DASK_NODES / squeue")
    print_info(f"Collecting {cfg.bench_count} benchmarks on {len(workers)} worker nodes ({len(groups[0])} per node)...")

    weights = model.state_dict()
    exec_data_file = fl.exec_data_file
    traj_start = time()
    futures = []
    for group_idx, group in enumerate(groups):
        worker = workers[group_idx % len(workers)]
        futures.append(dm.client.submit(
            rollout_group, group_idx, group, weights, step, exec_data_file,
            workers=[worker], resources={'single_task_slot': 1}, pure=False
        ))

    try:
        from distributed import wait
        done, not_done = wait(futures, timeout=GROUP_TIMEOUT)
    except TimeoutError:
        not_done = futures
        done = []
    if not_done:
        print_error(f"{len(not_done)} group task(s) did not finish within {GROUP_TIMEOUT}s — cancelling")
        dm.client.cancel(list(not_done), reason='group-timeout', msg='Group task timed out', force=True)
    gathered = dm.client.gather(list(done), errors='skip')
    # Surface task errors (e.g. worker-side deserialization failures) instead of
    # silently skipping the group — a systematic failure must be visible.
    for f in done:
        if f.status == 'error':
            try:
                f.result()
            except Exception as e:
                print_error(f"Group task errored: {type(e).__name__}: {e}")

    new_cache_data: dict[str, dict[str, int]] = {}
    all_entropies: list[float] = []
    all_speedups: list[float] = []
    segments: list[TrajectoryCollector] = []
    for group_idx, (segment, cache, entropies, speedups) in gathered:
        segments.append(segment)
        new_cache_data.update(cache)
        all_entropies.extend(entropies)
        all_speedups.extend(speedups)
    failed_groups = len(groups) - len(segments)
    if failed_groups:
        print_error(f"{failed_groups} group(s) failed this iteration — their benchmarks are skipped")
    if not segments:
        raise RuntimeError("All distributed groups failed — check worker logs (imports, PYTHONPATH, conda env)")

    tc = sum(segments, TrajectoryCollector())
    fl['train/entropy'].extend(all_entropies)
    fl['train/reward'].extend(tc.rewards)
    fl['train/final_speedup'].extend(all_speedups)
    exe.update_execution_cache(new_cache_data)

    # Surface benchmark failures instead of letting them pass silently
    total_dispatched = sum(len(g) for g in groups)
    bench_failures = total_dispatched - sum(len(v) for v in new_cache_data.values())
    if bench_failures > total_dispatched // 2:
        print_alert(f"{bench_failures}/{total_dispatched} benchmarks failed this iteration — possible silent infrastructure issue")
    elif bench_failures:
        print_info(f"{bench_failures} benchmark failures this iteration")

    print_info(f"Distributed collection: {timedelta(seconds=time() - traj_start)} for {len(tc.rewards)} transitions ({bench_failures} bench failures)")
    return tc.to_trajectory()
