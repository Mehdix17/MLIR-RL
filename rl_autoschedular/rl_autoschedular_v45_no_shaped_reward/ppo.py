import json
import os
from datetime import timedelta
from statistics import mean
import sys
import torch
from rl_autoschedular_v45_no_shaped_reward.env import Env
from rl_autoschedular_v45_no_shaped_reward.model import HiearchyModel as Model
from rl_autoschedular_v45_no_shaped_reward.state import OperationState
from rl_autoschedular_v45_no_shaped_reward.trajectory import TrajectoryCollector, TrajectoryData
from rl_autoschedular_v45_no_shaped_reward.observation import Observation, NumLoops
from rl_autoschedular_v45_no_shaped_reward.actions import ActionSpace
from rl_autoschedular_v45_no_shaped_reward.benchmarks import Benchmarks
from rl_autoschedular_v45_no_shaped_reward.execution import Execution
from rl_autoschedular_v1 import device
from utils.config import Config
from utils.file_logger import FileLogger
from utils.log import print_error, print_info, print_success
from utils.dask_manager import DaskManager
from time import time
from typing import Optional


def collect_trajectory(data: Benchmarks, model: Model, step: int):
    """Collect a trajectory using the model and the environment.

    Args:
        model (MyModel): The model to use.
        env (Env): The environment to use.
        step (int): The current step of the main loop
        tmp_exec_data_file (str): The path to the temporary execution data file.

    Returns:
        TrejectoryData: The collected trajectory.
    """
    dm = DaskManager()
    fl = FileLogger()
    exe = Execution()
    cfg = Config()

    eps = None
    if 'epsilon' in cfg.exploration:
        ratio = step / cfg.nb_iterations
        final_eps = 0.001
        eps = final_eps + (cfg.init_epsilon - final_eps) * (1 - ratio)

    print_info(f"Trajectory collection using {dm.num_workers} workers...")
    traj_start = time()

    # Prepare benchmarks to explore
    indices = torch.randperm(len(data))[:cfg.bench_count].long().tolist()
    if len(indices) < cfg.bench_count:
        indices = (indices * cfg.bench_count)[:cfg.bench_count]
    envs: list[Env] = []
    states: list[OperationState] = []
    observations: list[torch.Tensor] = []
    tcs: list[TrajectoryCollector] = []
    for idx in indices:
        env = Env()
        state = env.reset(data, idx)
        envs.append(env)
        states.append(state)
        observations.append(Observation.from_state(state))
        tcs.append(TrajectoryCollector())

    while (active_states := [(i, s) for i, s in enumerate(states) if not s.terminal]):
        # Sample states that are not terminal yet
        obss = torch.cat([observations[i] for i, _ in active_states])
        actions_index, actions_bev_log_p, entropies = model.sample(obss.to(device), eps=eps)
        actions_index, actions_bev_log_p, entropies = actions_index.cpu(), actions_bev_log_p.cpu(), entropies.cpu()
        fl['train/entropy'].extend(entropies.tolist())

        # Record data and update states
        for (i, state), obs, action_index, action_bev_log_p in zip(active_states, obss, actions_index, actions_bev_log_p):
            obs = obs.unsqueeze(0)

            # Get action and use it to get next state
            action = ActionSpace.action_by_index(action_index, state)
            states[i] = next_state = envs[i].step(state, action)
            observations[i] = next_obs = Observation.from_state(next_state)

            # If the benchmark is not done yet, keep next operation state instead
            done = False
            if next_state.terminal:
                next_op_state = envs[i].get_next_op_state(next_state)
                if next_op_state is not None:
                    states[i] = next_op_state
                    observations[i] = Observation.from_state(next_op_state)
                else:
                    done = True

            # Record available data
            tcs[i].append((
                Observation.get_part(obs, NumLoops).long().item(),
                action_index.unsqueeze(0),
                obs,
                next_obs,
                action_bev_log_p.item(),
                0.0,  # This will be filled after execution
                done
            ))

    traj_end_sampling = time()

    results = dm.map_states(__execute_states, states, data, exe.main_exec_data, training=True)

    traj_end_exec_states = time()

    results = [
        (*e.failed_seq(s.transformation_history), float(dm.batch_timeout))
        if not r else r
        for r, s, e in zip(results, states, envs)
    ]

    all_rewards, all_speedups, all_exec_times, cache_misses, worker_times = tuple(zip(*results))
    new_cache_data: dict[str, dict[str, int]] = {}
    for tc, state, rewards, speedup, exec_time in zip(tcs, states, all_rewards, all_speedups, all_exec_times):
        # Update trajectory
        # Safety Check: ensure rewards list length matches trajectory steps exactly
        if len(rewards) != len(tc.obs):
            print_error(f"Reward length mismatch for {state.bench_name}: rewards={len(rewards)}, steps={len(tc.obs)}. Fixing...")
            if len(rewards) > len(tc.obs):
                rewards = rewards[:len(tc.obs)]
            else:
                rewards.extend([0.0] * (len(tc.obs) - len(rewards)))

        tc.rewards = rewards
        # Log metrics
        fl['train/reward'].extend(rewards)
        fl['train/final_speedup'].append(speedup)
        # Get new cache data
        if exec_time is not None:
            cache_key = exe.get_code_cache_key(state.transformation_history)
            if state.bench_name not in new_cache_data:
                new_cache_data[state.bench_name] = {}
            new_cache_data[state.bench_name][cache_key] = exec_time

    tc = sum(tcs, TrajectoryCollector())
    # Save eval exec times as JSON (bench_name -> optimized_time_ns)
    eval_json: dict[str, Optional[int]] = {}
    for state, exec_time in zip(states, all_exec_times):
        eval_json[state.bench_name] = exec_time
    eval_dir = os.path.join(fl.logs_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, 'eval_exec_times.json'), 'w') as f:
        json.dump(eval_json, f, indent=2)
    exe.update_execution_cache(new_cache_data)

    traj_end = time()

    sampling_time = traj_end_sampling - traj_start
    exec_states_time = traj_end_exec_states - traj_end_sampling
    collection_time = traj_end - traj_start
    cache_miss_rate = mean(cache_misses) * 100
    sequential_time = sum(worker_times)
    distribted_speedup = sequential_time / exec_states_time
    perfect_speedup = sequential_time / max(worker_times)
    print_info((
        f"collection: {timedelta(seconds=collection_time)}"
        f", sampling: {timedelta(seconds=sampling_time)}"
        f", exec: {timedelta(seconds=exec_states_time)}\n"
        f"speedup: {distribted_speedup:.2f}x"
        f", perfect speedup: {perfect_speedup:.2f}x"
        f", cache miss rate: {cache_miss_rate:.2f}%"
    ))

    # Clear markers granularly after successful iteration to avoid race conditions
    marker_dir = os.path.join(cfg.results_dir, 'global_markers')
    for state in states:
        marker_file = os.path.join(marker_dir, state.bench_name)
        if os.path.exists(marker_file):
            try:
                os.remove(marker_file)
            except OSError:
                pass

    return tc.to_trajectory()


def ppo_update(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer):
    """Update the model using PPO.

    Args:
        trajectory (TrajectoryData): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.

    Returns:
        float: The average loss.
    """
    fl = FileLogger()
    cfg = Config()

    trajectory.update_attributes(model)

    ppo_start = time()
    data_loader = trajectory.loader(cfg.ppo_batch_size, 1)
    for _ in range(cfg.ppo_epochs):
        for batch in data_loader:
            batch: list[torch.Tensor] = [e.to(device, non_blocking=True) for e in batch]
            (
                _,
                actions_index,
                obs,
                _,
                actions_bev_log_p,
                _, _,
                values,
                _,
                actions_old_log_p,
                off_policy_rates,
                returns,
                advantages,
            ) = batch
            max_abs_adv = advantages.abs().max()
            if cfg.normalize_adv == 'standard' and advantages.size(0) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif cfg.normalize_adv == 'max-abs' and max_abs_adv > 0:
                advantages = advantages / max_abs_adv

            with torch.enable_grad():
                actions_log_p, new_values, entropies = model(obs, actions_index)

                policy_loss, clip_frac = model.policy_model.loss(actions_log_p, actions_bev_log_p, off_policy_rates, advantages)
                loss = policy_loss

                if cfg.value_epochs == 0:
                    value_loss = model.value_model.loss(new_values, values, returns)
                    loss += cfg.value_coef * value_loss

                if 'entropy' in cfg.exploration:
                    entropy_loss = -entropies.mean()
                    loss += cfg.entropy_coef * entropy_loss

            approx_kl = (actions_old_log_p - actions_log_p).pow(2).mean() / 2

            optimizer.zero_grad()
            try:
                loss.backward()
                clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print_error(
                    'Error during PPO update\n'
                    f'Error: {e}'
                )

            # Logging
            fl['train_ppo/policy_loss'].append(policy_loss.item())
            fl['train_ppo/clip_frac'].append(clip_frac.item())
            fl['train_ppo/clip_factor'].append(clip_factor.item())
            fl['train_ppo/approx_kl'].append(approx_kl.item())
            if cfg.value_epochs == 0:
                fl['train_ppo/value_loss'].append(value_loss.item())
            if 'entropy' in cfg.exploration:
                fl['train_ppo/entropy_loss'].append(entropy_loss.item())
    ppo_end = time()
    print_info(f"PPO fit in {timedelta(seconds=ppo_end - ppo_start)}")


def value_update(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer):
    """Update the value model using the trajectory.

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
    """
    fl = FileLogger()
    cfg = Config()

    trajectory.update_attributes(model)

    value_start = time()
    data_loader = trajectory.loader(cfg.value_batch_size, 1)
    for _ in range(cfg.value_epochs):
        for batch in data_loader:
            batch: list[torch.Tensor] = [e.to(device, non_blocking=True) for e in batch]
            (
                _, _,
                obs,
                _, _, _, _,
                values,
                _, _, _,
                returns,
                _,
            ) = batch
            with torch.enable_grad():
                new_values = model.value_model(obs)

                loss = model.value_model.loss(new_values, values, returns)

            optimizer.zero_grad()
            try:
                loss.backward()
                clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print_error(
                    'Error during Value update\n'
                    f'Error: {e}'
                )

            # Logging
            fl['train_value/loss'].append(loss.item())
            fl['train_value/clip_factor'].append(clip_factor.item())
    value_end = time()
    print_info(f"Value fit in {timedelta(seconds=value_end - value_start)}")


def evaluate_benchmarks(model: Model, data: Benchmarks):
    """Evaluate the benchmark using the model.

    Args:
        model (Model): The model to use.
        env (Env): The environment to use.
        tmp_exec_data_file (str): The path to the temporary execution data file.
    """
    dm = DaskManager()
    fl = FileLogger()
    exe = Execution()

    print_info("Evaluation started...")
    eval_start = time()

    # Prepare benchmarks to explore
    indices = range(len(data))
    if os.getenv("EVAL_BENCH_LAST_ONLY", "").strip().lower() in ("1", "true", "yes"):
        indices = [len(data) - 1]
        print_info(f"EVAL_BENCH_LAST_ONLY: evaluating only benchmark {data[indices[0]].bench_name}")

    envs: list[Env] = []
    states: list[OperationState] = []
    observations: list[torch.Tensor] = []
    for idx in indices:
        env = Env()
        state = env.reset(data, idx)
        envs.append(env)
        states.append(state)
        observations.append(Observation.from_state(state))

    while (active_states := [(i, s) for i, s in enumerate(states) if not s.terminal]):
        # Sample states that are not terminal yet
        obss = torch.cat([observations[i] for i, _ in active_states])
        actions_index, _, entropies = model.sample(obss.to(device), greedy=True)
        actions_index, entropies = actions_index.cpu(), entropies.cpu()
        fl['eval/entropy'].extend(entropies.tolist())

        # Record data and update states
        for (i, state), obs, action_index in zip(active_states, obss, actions_index):
            obs = obs.unsqueeze(0)

            # Get action and use it to get next state
            action = ActionSpace.action_by_index(action_index, state)
            states[i] = next_state = envs[i].step(state, action)
            observations[i] = Observation.from_state(next_state)

            # If the benchmark is not done yet, keep next operation state instead
            if next_state.terminal:
                next_op_state = envs[i].get_next_op_state(next_state)
                if next_op_state is not None:
                    states[i] = next_op_state
                    observations[i] = Observation.from_state(next_op_state)

    results = dm.map_states(__execute_states, states, data, exe.main_exec_data, training=False)
    results = [
        (*e.failed_seq(s.transformation_history), float(dm.batch_timeout))
        if not r else r
        for r, s, e in zip(results, states, envs)
    ]
    all_rewards, all_speedups, all_exec_times, _, _ = tuple(zip(*results))
    new_cache_data: dict[str, dict[str, int]] = {}
    for state, rewards, speedup, exec_time in zip(states, all_rewards, all_speedups, all_exec_times):
        fl['eval/reward'].extend(rewards)
        fl['eval/cumulative_reward'].append(sum(rewards))
        fl['eval/final_speedup'].append(speedup)
        if exec_time is not None:
            fl[f'eval/exec_time/{state.bench_name}'].append(exec_time)
            fl[f'eval/speedup/{state.bench_name}'].append(speedup)
            cache_key = exe.get_code_cache_key(state.transformation_history)
            if state.bench_name not in new_cache_data:
                new_cache_data[state.bench_name] = {}
            new_cache_data[state.bench_name][cache_key] = exec_time

        print_info("Bench: " + state.bench_name, add_label=False)
        print_info(str(state.transformation_history), add_label=False)

    if len(all_speedups) > 0:
        fl['eval/average_speedup'].append(sum(all_speedups) / len(all_speedups))
    exe.update_execution_cache(new_cache_data)

    eval_end = time()
    print_info(f"Evaluation time: {timedelta(seconds=eval_end - eval_start)}")

    # Clear markers granularly after successful evaluation to avoid race conditions
    marker_dir = os.path.join(Config().results_dir, 'global_markers')
    for state in states:
        marker_file = os.path.join(marker_dir, state.bench_name)
        if os.path.exists(marker_file):
            try:
                os.remove(marker_file)
            except OSError:
                pass


def __execute_states(state: OperationState, exec_data_file: str, benchs: Benchmarks, main_exec_data: Optional[dict[str, dict[str, int]]]):
    print(f"Handling bench: {state.bench_name}...", end=' ', file=sys.stderr)
    worker_start = time()

    # Check if this benchmark was already completed in a previous crashed run
    import json as _json
    # Use a persistent marker directory across runs for the same experiment
    cfg = Config()
    marker_dir = os.path.join(cfg.results_dir, 'global_markers')
    # Robust directory creation (handles race conditions)
    if not os.path.exists(marker_dir):
        try:
            os.makedirs(marker_dir, exist_ok=True)
        except OSError:
            pass

    marker_file = os.path.join(marker_dir, state.bench_name)
    if os.path.exists(marker_file):
        with open(marker_file) as f:
            cached = _json.load(f)
        print('Skipped (cached)', file=sys.stderr)
        return cached['rewards'], cached['speedup'], cached['exec_time'], cached.get('cache_miss', True), time() - worker_start

    Execution(exec_data_file, main_exec_data)
    env = Env()
    env.reset(benchs, state.bench_idx)
    rewards, speedup, new_exec_time, cache_miss = env.apply_and_run_sequence(state.transformation_history)

    # Save result atomically so crash-resilient
    import os as _os
    tmp_file = marker_file + f'.tmp.{_os.getpid()}'
    # Ensure dir exists before writing tmp
    tmp_dir = _os.path.dirname(tmp_file)
    if not os.path.exists(tmp_dir):
        try:
            os.makedirs(tmp_dir, exist_ok=True)
        except OSError:
            pass

    with open(tmp_file, 'w') as f:
        _json.dump({'rewards': rewards, 'speedup': speedup, 'exec_time': new_exec_time, 'cache_miss': cache_miss}, f)
    os.rename(tmp_file, marker_file)

    worker_end = time()
    worker_time = worker_end - worker_start
    print('Done', file=sys.stderr)

    return rewards, speedup, new_exec_time, cache_miss, worker_time
