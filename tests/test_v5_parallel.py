"""Unit tests for V5.1 distributed/parallel training helpers.

Run with:  python -m unittest tests.test_v5_parallel -v
No external deps (stdlib unittest). MLIR-dependent tests skip when the env
vars are missing. The env must be sourced first for the env-dependent tests:
  source ~/envs/mlir/bin/activate && set -a && source .env && set +a
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'utils'))

import report_parallel  # noqa: E402

# Hyperparameter keys shared between the v5 T6 configs and the paper config.
PAPER_PARITY_KEYS = [
    'max_num_stores_loads', 'max_num_loops', 'max_num_load_store_dim',
    'num_tile_sizes', 'num_pad_multiples', 'num_unroll_factors', 'vect_size_limit',
    'order', 'interchange_mode', 'exploration', 'init_epsilon', 'normalize_bounds',
    'normalize_adv', 'reuse_experience', 'replay_count', 'nb_iterations',
    'ppo_epochs', 'ppo_batch_size', 'value_epochs', 'value_batch_size',
    'value_clip', 'value_coef', 'entropy_coef', 'truncate', 'debug',
    'main_exec_data_file', 'ppo_clip_range', 'gae_lambda', 'max_grad_norm',
    'bench_count', 'lr',
]


class TestReportParallel(unittest.TestCase):
    def test_to_secs(self):
        self.assertAlmostEqual(report_parallel._to_secs('0:06:15.97'), 375.97)
        self.assertAlmostEqual(report_parallel._to_secs('12.5'), 12.5)
        self.assertAlmostEqual(report_parallel._to_secs('0:00:29.44'), 29.44)

    def test_parse_log(self):
        log = (
            '08-13 17:02 - [INFO]    - Main Loop 2/50 (4.00%) (0/it) (0 < 0)\n'
            '08-13 17:03 - [INFO]    Collecting 64 benchmarks on 4 worker nodes (4 per node)...\n'
            '08-13 17:09 - [INFO]    Distributed collection: 0:06:15.97 for 1624 transitions\n'
            '08-13 17:09 - [INFO]    PPO fit in 0:00:12.59\n'
            '08-13 17:15 - [INFO]    Distributed collection: 0:00:15.05 for 1541 transitions\n'
            '08-13 17:15 - [INFO]    - Main Loop 13/50 (26.00%) (0:00:29.44/it) (0:12:29 < 0:39:34)\n'
            '08-13 17:16 - [INFO]    - Main Loop 14/50 (28.00%) (0:00:28.14/it) (0:12:58 < 0:36:54)\n'
        )
        with tempfile.NamedTemporaryFile('w', suffix='.out', delete=False) as f:
            f.write(log)
            path = f.name
        try:
            walls, colls, fits, trans = report_parallel.parse_log(path)
        finally:
            os.unlink(path)
        self.assertEqual(walls, [29.44, 28.14])
        self.assertEqual(colls, [375.97, 15.05])
        self.assertEqual(fits, [12.59])
        self.assertEqual(trans, [1624, 1541])


class TestConfigs(unittest.TestCase):
    def test_paper_hyperparameter_parity(self):
        paper = json.load(open(REPO / 'config/ops_and_blocks/train/paper_original.json'))
        for cfg_name in ('v5_single_node.json', 'v5_distributed.json', 'v5_no_transformer.json'):
            cfg = json.load(open(REPO / 'config' / 'v5' / cfg_name))
            for key in PAPER_PARITY_KEYS:
                self.assertEqual(
                    cfg.get(key), paper.get(key),
                    f'{cfg_name}: {key} differs from paper_original.json '
                    f'({cfg.get(key)} != {paper.get(key)})')

    def test_distributed_config_seed_and_results_dir(self):
        cfg = json.load(open(REPO / 'config' / 'v5' / 'v5_distributed.json'))
        self.assertEqual(cfg['seed'], 42)
        self.assertEqual(cfg['results_dir'], 'results/ops_and_blocks_results/v5_distributed_agent')
        self.assertEqual(cfg['implementation'], 'rl_autoschedular_v5')
        self.assertEqual(cfg['bench_count'], 64)
        self.assertEqual(cfg['dask_node_count'], 16)
        self.assertEqual(cfg['dask_worker_cores'], 8)
        self.assertEqual(cfg['dask_worker_mem'], '3GB')


class TestSeed(unittest.TestCase):
    def _env(self):
        inherited = os.environ.get('PYTHONPATH', '')
        env = {**os.environ,
               'PYTHONPATH': f'{REPO}:{REPO}/rl_autoschedular:{inherited}',
               'CONFIG_FILE_PATH': str(REPO / 'config' / 'v5' / 'v5_single_node.json')}
        return env

    @unittest.skipUnless(os.environ.get('LLVM_BUILD_PATH'), 'MLIR env not sourced')
    def test_seed_reproduces_model_init(self):
        code = (
            'import torch\n'
            'from rl_autoschedular_v5.model import HiearchyModel\n'
            'def build(seed):\n'
            '    torch.manual_seed(seed)\n'
            '    return {k: v.clone() for k, v in HiearchyModel().state_dict().items()}\n'
            'a = build(42); b = build(42)\n'
            'print(all(torch.equal(a[k], b[k]) for k in a))\n'
        )
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           env=self._env())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], 'True')

    def test_config_loads_seed_via_singleton(self):
        code = (
            'import os; os.environ["CONFIG_FILE_PATH"] = os.environ["_CFG"]\n'
            'from utils.config import Config\n'
            'c = Config()\n'
            'print(c.seed, c.max_num_loops, c.bench_count)\n'
        )
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                           env={**self._env(), '_CFG': str(REPO / 'config' / 'v5' / 'v5_distributed.json')})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.strip().splitlines()[-1], '42 7 64')


if __name__ == '__main__':
    unittest.main()
