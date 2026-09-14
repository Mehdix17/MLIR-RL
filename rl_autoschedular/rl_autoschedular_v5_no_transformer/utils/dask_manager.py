"""Distributed computation management using Dask.

This module handles distributed parallel execution of benchmark evaluations
across multiple worker nodes. It provides abstractions for mapping functions
across data in a distributed manner with resource management.
"""

import os
from time import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

from .file_logger import FileLogger
from .singleton import Singleton
from .log import print_alert, print_error, print_info, print_success

if TYPE_CHECKING:
    from rl_autoschedular_v5_no_transformer.benchmarks import Benchmarks
    from distributed import Future

def _dask_node_count() -> int:
    """Number of dask workers: env DASK_NODES > config dask_node_count > 0."""
    env = os.getenv('DASK_NODES')
    if env is not None:
        return int(env)
    try:
        from rl_autoschedular_v5_no_transformer.utils.config import Config
        return int(getattr(Config(), 'dask_node_count', 0) or 0)
    except Exception:
        return 0


ENABLED = _dask_node_count() > 0 and os.getenv('DASK_TRAINING_ONLY') != '1'
T = TypeVar('T')
obj_T = TypeVar('obj_T')


class DaskManager(metaclass=Singleton):
    """DaskManager class for distributed parallel execution."""

    def __init__(self):
        if not ENABLED:
            return

        import shutil

        from dask_jobqueue import SLURMCluster
        from dask_jobqueue.slurm import SLURMJob
        from distributed import Client

        # Resolve slurm CLI binaries to absolute paths: compute nodes don't all
        # expose /opt/slurm/default/bin on PATH, and dask-jobqueue execs the bare
        # 'sbatch' string (class attr) which then fails with ENOENT.
        def _find_slurm_bin(name: str) -> str:
            found = shutil.which(name)
            if found:
                return found
            for prefix in ('/opt/slurm/default/bin', '/opt/slurm/20.11.4-13/bin', '/usr/bin'):
                candidate = os.path.join(prefix, name)
                if os.path.exists(candidate):
                    return candidate
            return name

        self.sbatch = _find_slurm_bin('sbatch')
        self.scancel = _find_slurm_bin('scancel')
        SLURMJob.submit_command = self.sbatch
        SLURMJob.cancel_command = self.scancel
        print_info(f"Using slurm binaries: sbatch={self.sbatch}, scancel={self.scancel}")

        # Worker PYTHONPATH, built deterministically: the driver's PYTHONPATH env
        # var can carry a literal $PYTHONPATH (sourced from .env) which would
        # expand to nothing in the worker's bare env.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))  # .../utils -> MLIR-RL
        worker_pythonpath = (
            f"{os.getenv('LLVM_BUILD_PATH', '')}/tools/mlir/python_packages/mlir_core"
            f":{project_root}:{project_root}/rl_autoschedular"
        )
        conda_bin = (os.getenv('CONDA_ENV') or os.path.expanduser('~/envs/mlir/bin/activate')).replace('/bin/activate', '/bin')
        worker_path = f"/usr/local/bin:/usr/bin:/bin:/opt/slurm/default/bin:{conda_bin}"
        print_info(f"Driver PYTHONPATH={os.getenv('PYTHONPATH', '')!r}")
        print_info(f"Worker PYTHONPATH={worker_pythonpath}")
        print_info(f"Worker PATH={worker_path}")

        enable_dashboard = True
        dask_reservation = os.getenv('DASK_RESERVATION')
        # CONDA_ENV points at the activate script (~/envs/mlir/bin/activate);
        # conda activate wants the env dir.
        dask_conda_env = (os.getenv('CONDA_ENV') or '').replace('/bin/activate', '')
        # Worker sizing: env override > JSON config > measured defaults
        from rl_autoschedular_v5_no_transformer.utils.config import Config
        _cfg = Config()
        _cores = getattr(_cfg, 'dask_worker_cores', None)
        _mem = getattr(_cfg, 'dask_worker_mem', None)
        _excl = getattr(_cfg, 'dask_worker_exclusive', None)
        worker_cores = int(os.getenv('DASK_WORKER_CORES', str(_cores or 8)))
        worker_mem = os.getenv('DASK_WORKER_MEM', _mem or '3GB')
        worker_exclusive = int(os.getenv('DASK_WORKER_EXCLUSIVE', '1' if _excl else '0'))
        cluster = SLURMCluster(
            job_name='dask',
            queue='compute',
            cores=worker_cores,   # measured: ~4 parallel MLIR execs + policy fwd; 8c covers bursts
            processes=1,
            nanny=True,
            memory=worker_mem,    # measured: ~0.7GB RSS/worker; 2GB = 3x safety
            walltime='7-00',
            shebang='#!/bin/bash',
            job_extra_directives=[
                f'--reservation={dask_reservation}' if dask_reservation else '',
                '--nodes=1',
                '--constraint=bergamo',
                '--exclusive' if worker_exclusive > 0 else '',  # lean default: no exclusivity
            ],
            worker_extra_args=['--resources', 'single_task_slot=1'],
            log_directory='dask-logs',
            job_script_prologue=[
                # Slurm batch jobs on this cluster do NOT inherit the driver env
                # (bare PATH); export everything the worker path needs. The
                # worker command itself uses the absolute env python, so no
                # module/conda activation is required.
                *(f'export {v}={os.getenv(v, "")}' for v in (
                    'LD_LIBRARY_PATH', 'LLVM_BUILD_PATH',
                    'AST_DUMPER_BIN_PATH', 'VECTORIZER_BIN_PATH',
                    'CONFIG_FILE_PATH', 'AUTOSCHEDULER_IMPL')),
                f'export PATH={worker_path}',
                f'export PYTHONPATH={worker_pythonpath}',
                'export OMP_NUM_THREADS=12',
                # Nanny spawns workers as daemonic by default; execution.py spawns a
                # multiprocessing.Process for MLIR isolation, which daemonic processes
                # cannot do ("daemonic processes are not allowed to have children").
                'export DASK_DISTRIBUTED__WORKER__DAEMON=False',
            ],
            scheduler_options={
                'dashboard': enable_dashboard,
                'worker_ttl': '3600s'
            }
        )
        self.cluster = cluster

        num_nodes_to_use = _dask_node_count()
        print_info(f"Requesting {num_nodes_to_use} nodes for Dask workers...")
        cluster.scale(jobs=num_nodes_to_use)

        # Wait for the scheduler (jobs submitted), then connect by address.
        # Client(cluster) triggers SpecCluster state-correction, which races with
        # fresh job creation and immediately closes the submitted worker jobs.
        async def _wait_scheduler():
            await cluster
        cluster.sync(_wait_scheduler)

        client = Client(cluster.scheduler_address)
        self.client = client
        print_success("Dask client connected!", f"  Dashboard at: {client.dashboard_link}" if enable_dashboard else "")

        self.__keep_only_running()
        print_success(f"Got {self.num_workers} nodes")

        self.batch_timeout = 300
        self.persistent_funcs: dict[str, Callable[[], Any]] = {}
        self.persistent_futures: dict[str, 'Future'] = {}

    @property
    def workers_names(self) -> list[str]:
        """Names of workers currently CONNECTED to the scheduler.

        Only connected workers get tasks — job names from cluster.workers can
        include not-yet-connected (or dead) jobs, which would strand tasks.
        """
        if not ENABLED:
            return []
        # n_workers=-1: scheduler_info defaults to 5 workers only — without -1 we'd
        # silently dispatch to a 5-worker subset of the cluster.
        return [w['name'] for w in self.client.scheduler_info(n_workers=-1).get('workers', {}).values() if w.get('name')]

    @property
    def num_workers(self) -> int:
        """Number of workers currently connected to the scheduler."""
        if not ENABLED:
            return 0
        return len(self.workers_names)

    def map_objs(
        self,
        func: Callable[[obj_T, str, 'Benchmarks', Optional[dict[str, dict[str, int]]]], T],
        objs: Iterable[obj_T],
        benchs: 'Benchmarks',
        main_exec_data: Optional[dict[str, dict[str, int]]],
        training: bool,
        obj_str: Callable[[obj_T], str] = lambda o: str(o)
    ) -> list[Optional[T]]:
        """Map a function across objects in a distributed manner.

        Args:
            func: The function to apply to each object.
            objs: The objects to apply the function to.
            benchs: The benchmark suite to use.
            main_exec_data: The main execution data (if available).
            training: Whether the mapping is for training. if True,
                the function will be executed with a timeout and the
                training benchmarks will be used instead of evaluation.
            obj_str: A function to convert each object to a string for logging.

        Returns:
            A list of the results of the function applied to each object.
        """

        if not ENABLED or self.num_workers == 0:
            n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            if n_workers <= 1:
                return [func(o, FileLogger().exec_data_file, benchs, main_exec_data) for o in objs]
            import concurrent.futures
            exec_file = FileLogger().exec_data_file
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(func, o, exec_file, benchs, main_exec_data) for o in objs]
                return [f.result() for f in futures]

        from distributed import as_completed

        # Prepare objs for submission
        objs_count = len(objs)
        ordered_objs = list(zip(range(objs_count), objs))
        results: list[Optional[T]] = [None] * objs_count
        future_to_worker: dict['Future', str] = {}

        # Submit first objs to each worker
        initial_objs_count = min(objs_count, self.num_workers)
        for i in range(initial_objs_count):
            worker_name = self.workers_names[i]
            future = self.__submit_obj(func, *ordered_objs.pop(0), worker_name, training)
            future_to_worker[future] = worker_name

        # Process futures as they finish
        ac = as_completed(future_to_worker.keys(), with_results=True, timeout=self.batch_timeout if training else None)
        try:
            for future, indexed_result in ac:
                future: 'Future'
                indexed_result: tuple[int, T]

                idx, result = indexed_result
                results[idx] = result
                freed_worker = future_to_worker.pop(future)

                # If there are still remaining objs submit them
                if ordered_objs:
                    new_future = self.__submit_obj(func, *ordered_objs.pop(0), freed_worker, training)
                    future_to_worker[new_future] = freed_worker

                    # Include the new future in the queue
                    ac.add(new_future)

        except TimeoutError:
            self.client.cancel(list(future_to_worker.keys()), reason='task-timeout', msg='Task timed out', force=True)
            failed_workers = list(future_to_worker.values())
            try:
                self.client.restart_workers(failed_workers, raise_for_error=False)
            except Exception:
                pass
            restarted_workers = set(failed_workers).intersection(set(self.workers_names))
            unrestarted_workers = set(failed_workers) - set(self.workers_names)
            for worker in restarted_workers:
                self.__renew_worker_persistents(worker)
            print_error(
                "States exec timed out\n"
                f"Cancelling benchmarks: {[obj_str(o) for o, r in zip(objs, results) if r is None]}\n"
                f"Unvisited benchmarks: {[obj_str(o) for _, o in ordered_objs]}\n"
                f"Restarted workers: {restarted_workers}\n"
                f"Failed to restart workers: {unrestarted_workers}"
            )

        return results

    def run_and_register_to_workers(self, func: Callable[[], T]) -> T:
        """Run a function both locally and on the workers.
        The result will be registered to all workers, and
        returned by this function.

        Args:
            func: The function to run.

        Returns:
            The result of the function.
        """

        if not ENABLED or self.num_workers == 0:
            return func()

        key = func.__name__
        if key in self.persistent_funcs:
            return func()
        self.persistent_funcs[key] = func

        for worker in self.workers_names:
            self.__submit_persistent(key, worker)

        return func()

    def __submit_persistent(self, key: str, worker: str) -> 'Future':
        """Submit a persistent function to a worker,
        and keep track of its result (Future) for re-use.

        Args:
            key: The key of the function.
            worker: The worker to submit the function to.

        Returns:
            The future of the function.
        """
        assert key in self.persistent_funcs, f"Task {key} expected to be registered"
        func = self.persistent_funcs[key]

        worker_key = f'{key}_{worker}'
        assert worker_key not in self.persistent_futures, f"Future {key} was found existing in worker {worker}"

        future = self.client.submit(
            func,
            workers=worker,
            pure=False
        )
        self.persistent_futures[worker_key] = future

        return future

    def __get_persistent(self, key: str, worker: str) -> 'Future':
        """Get the result of a persistent function from a worker.

        Args:
            key: The key of the function.
            worker: The worker to get the result from.

        Returns:
            The future that points to the result of the function.
        """

        worker_key = f'{key}_{worker}'
        if worker_key in self.persistent_futures:
            return self.persistent_futures[worker_key]

        print_alert(f"Future {key} not found in worker {worker}, attemtping recomputation!")
        if key in self.persistent_funcs:
            return self.__submit_persistent(key, worker)

        raise Exception(f"Unable to find or compute future {key}")

    def __renew_persistent(self, key: str, worker: str) -> 'Future':
        """Recompute the result of a persistent function on a worker.
        This should be called when a persistent result (Future) has
        become invalid (due to a worker failure mostly).

        Args:
            key: The key of the function.
            worker: The worker to renew the result on.

        Returns:
            The future that points to the result of the function.
        """
        worker_key = f'{key}_{worker}'
        if worker_key in self.persistent_futures:
            del self.persistent_futures[worker_key]

        return self.__submit_persistent(key, worker)

    def __renew_worker_persistents(self, worker: str):
        """Recompute all persistent functions on a worker.
        This should be called when a worker has failed.

        Args:
            worker: The worker to renew the results on.
        """
        for key in self.persistent_funcs:
            self.__renew_persistent(key, worker)

    def __submit_obj(
        self,
        func: Callable[[obj_T, str, 'Benchmarks', Optional[dict[str, dict[str, int]]]], T],
        idx: int,
        obj: obj_T,
        worker: str,
        training: bool
    ) -> 'Future':
        """Execute a function on an object, and submit it to a worker.

        Args:
            func: The function to execute.
            idx: The index of the object (for tracking purposes).
            obj: The object to execute the function on.
            worker: The worker to submit the result to.
            training: Whether the object is for training. if True,
                the function will be executed with a timeout and the
                training benchmarks will be used instead of evaluation.

        Returns:
            The future that points to the result of the function.
        """
        # Add a wrapper to track state order
        def func_wrapper(idx: int, *args):
            return idx, func(*args)
        func_wrapper.__name__ = func.__name__ + '_wrapper'

        exec_data_file = FileLogger().exec_data_file
        benchs = self.__get_persistent('load_train_data' if training else 'load_eval_data', worker)
        main_exec_data = self.__get_persistent('load_main_exec_data', worker)

        return self.client.submit(
            func_wrapper,
            idx, obj, exec_data_file, benchs, main_exec_data,
            workers=worker,
            resources={'single_task_slot': 1},
            pure=False
        )

    def __keep_only_running(self):
        """Wait for workers to connect to the scheduler.

        Uses wait_for_workers (live scheduler state) — Client.scheduler_info()
        is a cached client-side view that can lag behind actual registrations,
        and a polling race previously caused healthy workers to be scancel'd.
        Workers that never connect are simply not used (workers_names returns
        connected workers only); nothing is killed.
        """
        workers: dict = self.cluster.workers
        if not workers:
            return
        print_info(f"Waiting for {len(workers)} workers to connect...")
        try:
            self.client.wait_for_workers(len(workers), timeout=600)
            print_success(f"All {len(workers)} workers connected")
        except Exception as e:
            print_alert(f"Only {len(self.workers_names)}/{len(workers)} workers connected within 600s ({type(e).__name__}); continuing with those")

    def close(self):
        """Close the cluster and client."""
        self.client.close()
        self.cluster.close()
