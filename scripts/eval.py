# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

# Import modules
import os
import logging
import torch
from typing import Optional
from utils.dask_manager import DaskManager
from utils.file_logger import FileLogger
from utils.config import Config
from utils.log import print_info, print_success
from utils.implementation import get_agent_runs_root, get_autoschedular_impl, import_autoschedular_module
from datetime import timedelta
from time import time

AUTOSCHEDULER_IMPL = get_autoschedular_impl()
Execution = import_autoschedular_module("execution", AUTOSCHEDULER_IMPL).Execution
Model = import_autoschedular_module("model", AUTOSCHEDULER_IMPL).HiearchyModel
device = import_autoschedular_module("", AUTOSCHEDULER_IMPL).device
evaluate_benchmarks = import_autoschedular_module("ppo", AUTOSCHEDULER_IMPL).evaluate_benchmarks
Benchmarks = import_autoschedular_module("benchmarks", AUTOSCHEDULER_IMPL).Benchmarks

logging.basicConfig(
    filename=f"logs/{os.getenv('SLURM_JOB_NAME', 'interactive')}_{os.environ['SLURM_JOB_ID']}.debug",
    filemode="w",
    format="${asctime} - [${levelname}]    ${name}: ${message}",
    datefmt="%m-%d %H:%M",
    style='$',
    level=logging.DEBUG
)

# Initialize singleton classes
cfg = Config()
fl = FileLogger()
dm = DaskManager()


# Data loading
def load_eval_data():
    return Benchmarks(is_training=False)


def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
    return None


eval_data = dm.run_and_register_to_workers(load_eval_data)
main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

# Initialize execution singleton
Execution(fl.exec_data_file, main_exec_data)

# Prepare logging
print_info(f"Config: {cfg}")
print_info(f"Autoscheduler implementation: {AUTOSCHEDULER_IMPL}")
print_success(f'Logging to: {fl.run_dir}')

# Setup torch
torch.set_grad_enabled(False)
torch.set_num_threads(4)

# Initiate model
model = Model().to(device)
print_success("Model initialized")

# Start evaluation
eval_dir = os.getenv('EVAL_DIR')
if eval_dir is None:
    # Derive from config: find the latest run_N for the selected implementation
    # whose models/ dir has .pt files.
    # (the current eval run's models/ dir is always empty)
    _agent_root = str(get_agent_runs_root(cfg.results_dir, AUTOSCHEDULER_IMPL))
    if not os.path.isdir(_agent_root):
        raise ValueError(
            "No implementation run directory found. "
            f"Expected: {_agent_root}. Run training first or set EVAL_DIR explicitly."
        )
    _runs = sorted(
        [d for d in os.listdir(_agent_root) if d.startswith('run_') and d.split('_')[-1].isdigit()],
        key=lambda x: int(x.split('_')[1])
    )
    _candidates = [
        os.path.join(_agent_root, d, 'models') for d in _runs
        if any(f.endswith('.pt') for f in os.listdir(os.path.join(_agent_root, d, 'models'))
               if os.path.isdir(os.path.join(_agent_root, d, 'models')))
    ]
    if not _candidates:
        raise ValueError(
            "No run with saved model checkpoints found in results_dir. "
            "Set EVAL_DIR explicitly or run training first. "
            f"Looked under: {_agent_root}"
        )
    eval_dir = _candidates[-1]
    print(f"EVAL_DIR not set; using: {eval_dir}")
eval_dir = os.path.abspath(eval_dir)

# Read the files in the evaluation directory
eval_files = [f for f in os.listdir(eval_dir) if f.endswith('.pt')]

# Order files
eval_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

iter_time_dlt = 0
elapsed_dlt = 0
eta_dlt = 0
overall_start = time()
models_count = len(eval_files)
for step, model_file in enumerate(eval_files):
    print_info(
        f"- Evaluation {step + 1}/{models_count}"
        f" ({100 * (step + 1) / models_count:.2f}%)"
        f" ({iter_time_dlt}/it) ({elapsed_dlt} < {eta_dlt})",
        flush=True
    )

    main_start = time()

    model_path = os.path.join(eval_dir, model_file)
    if not os.path.exists(model_path):
        print_info(f"Model file {model_path} does not exist. Skipping.")
        continue
    model.load_state_dict(torch.load(model_path, weights_only=True))

    evaluate_benchmarks(model, eval_data)

    main_end = time()
    iter_time = main_end - main_start
    elapsed = main_end - overall_start
    eta = elapsed * (cfg.nb_iterations - step - 1) / (step + 1)
    iter_time_dlt = timedelta(seconds=iter_time)
    elapsed_dlt = timedelta(seconds=int(elapsed))
    eta_dlt = timedelta(seconds=int(eta))
