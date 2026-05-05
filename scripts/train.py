from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
load_dotenv('.env.debug')

import os
import logging
import torch
import json
from utils.log import print_info, print_success
from utils.config import Config
from utils.dask_manager import DaskManager
from utils.file_logger import FileLogger
from utils.implementation import get_autoschedular_impl, import_autoschedular_module
from typing import Optional
from time import time
from datetime import timedelta

AUTOSCHEDULER_IMPL = get_autoschedular_impl()
Benchmarks = import_autoschedular_module("benchmarks", AUTOSCHEDULER_IMPL).Benchmarks
Execution = import_autoschedular_module("execution", AUTOSCHEDULER_IMPL).Execution
Model = import_autoschedular_module("model", AUTOSCHEDULER_IMPL).HiearchyModel
device = import_autoschedular_module("", AUTOSCHEDULER_IMPL).device
TrajectoryData = import_autoschedular_module("trajectory", AUTOSCHEDULER_IMPL).TrajectoryData
ppo_module = import_autoschedular_module("ppo", AUTOSCHEDULER_IMPL)
collect_trajectory = ppo_module.collect_trajectory
ppo_update = ppo_module.ppo_update
value_update = ppo_module.value_update

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
def load_train_data():
    return Benchmarks()




def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
    main_exec_data = None
    if Config().main_exec_data_file:
        with open(Config().main_exec_data_file) as f:
            main_exec_data = json.load(f)
    return main_exec_data


train_data = dm.run_and_register_to_workers(load_train_data)
main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

# Initialize execution singleton
Execution(fl.exec_data_file, main_exec_data)

print_info(f"Config: {cfg}")
print_info(f"Autoscheduler implementation: {AUTOSCHEDULER_IMPL}")
print_success(f'Logging to: {fl.run_dir}')
if cfg.main_exec_data_file:
    print_info(f"Global execution data located in: {cfg.main_exec_data_file}")

# Setup torch
torch.set_grad_enabled(False)
torch.set_num_threads(8)
if cfg.debug:
    torch.autograd.set_detect_anomaly(True)

# Initiate model
model = Model().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)
print_success("Model initialized")

# Check for existing models to resume training
start_step = 0
import re
models_list = [f for f in os.listdir(fl.models_dir) if f.startswith('model_') and f.endswith('.pt')]
if models_list:
    import builtins
    latest_model_file = builtins.max(models_list, key=lambda f: int(re.search(r'model_(\d+)\.pt', f).group(1)))
    latest_step = int(re.search(r'model_(\d+)\.pt', latest_model_file).group(1))
    model.load_state_dict(torch.load(os.path.join(fl.models_dir, latest_model_file), map_location=device, weights_only=True))
    start_step = latest_step + 1
    print_success(f"Resumed model from checkpoint {latest_model_file} (Start step: {start_step})")

# Start training
old_trajectory: Optional[TrajectoryData] = None
iter_time_dlt = 0
elapsed_dlt = 0
eta_dlt = 0
overall_start = time()
for step in range(start_step, cfg.nb_iterations):
    print_info(
        f"- Main Loop {step + 1}/{cfg.nb_iterations}"
        f" ({100 * (step + 1) / cfg.nb_iterations:.2f}%)"
        f" ({iter_time_dlt}/it) ({elapsed_dlt} < {eta_dlt})",
        flush=True
    )

    main_start = time()

    # Collect trajectory using the model
    trajectory = collect_trajectory(train_data, model, step)

    # Extend trajectory with previous trajectory
    if cfg.reuse_experience != 'none':
        reuse_start = time()
        if old_trajectory is not None:
            trajectory = old_trajectory + trajectory
        old_trajectory = trajectory.copy()
        reuse_end = time()
        reuse_time_ms = int((reuse_end - reuse_start) * 1000)
        print_info(f"Reuse time: {reuse_time_ms}ms")

    # Fit value model to trajectory rewards
    if cfg.value_epochs > 0:
        value_update(trajectory, model, optimizer)

    # Update policy model with PPO
    ppo_update(trajectory, model, optimizer)

    # Save the model
    if (step + 1) % 25 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(
                fl.models_dir,
                f'model_{step}.pt'
            )
        )



    main_end = time()
    iter_time = main_end - main_start
    elapsed = main_end - overall_start
    eta = elapsed * (cfg.nb_iterations - step - 1) / (step + 1)
    iter_time_dlt = timedelta(seconds=iter_time)
    elapsed_dlt = timedelta(seconds=int(elapsed))
    eta_dlt = timedelta(seconds=int(eta))


