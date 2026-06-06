from .singleton import Singleton
from .config import Config
from .implementation import get_agent_runs_root, get_autoschedular_impl
import json
import os


class FileLogger(metaclass=Singleton):
    """Class to log results to files"""
    def __init__(self):
        cfg = Config()
        self._tags = ['ppo'] + cfg.tags
        implementation = get_autoschedular_impl()

        # Agent root = results_dir directly (v4_7_agent/, etc.)
        self.run_dir = str(get_agent_runs_root(cfg.results_dir, implementation))
        os.makedirs(self.run_dir, exist_ok=True)

        # Safety: refuse to start fresh training if results already exist
        self._train_results = os.path.join(self.run_dir, "train", "results.json")
        resume_from = os.getenv("RESUME_FROM")
        force_new = os.getenv("FORCE_NEW")
        if (resume_from is None and force_new is None
                and os.path.isfile(self._train_results)
                and os.path.getsize(self._train_results) > 2):
            raise RuntimeError(
                f"Experiment at {self.run_dir} already has training results. "
                "Use --resume to continue training, or set FORCE_NEW=1 to overwrite."
            )

        self._setup_dirs()

    def _setup_dirs(self):
        tags_file = os.path.join(self.run_dir, 'tags')
        with open(tags_file, 'w') as f:
            f.write('\n'.join(self._tags))
            f.write('\n')

        self.models_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.logs_dir = os.path.join(self.run_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)

        self.exec_data_file = os.path.join(self.logs_dir, 'exec_data.json')
        if not os.path.exists(self.exec_data_file):
            with open(self.exec_data_file, "w") as f:
                json.dump({}, f)

        self.train_dir = os.path.join(self.run_dir, 'train')
        os.makedirs(self.train_dir, exist_ok=True)
        self.train_results_file = os.path.join(self.train_dir, 'results.json')
        if not os.path.exists(self.train_results_file):
            with open(self.train_results_file, "w") as f:
                json.dump({}, f)

        self.eval_dir = os.path.join(self.run_dir, 'eval')
        os.makedirs(self.eval_dir, exist_ok=True)

        self.files_dict: dict[str, FileInstance] = {}

    def __getitem__(self, path: str):
        if path not in self.files_dict:
            full_path = os.path.join(self.logs_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            self.files_dict[path] = FileInstance(full_path)
        return self.files_dict[path]


class FileInstance:
    def __init__(self, path: str):
        self.path = path

    def append(self, data):
        with open(self.path, 'a') as f:
            f.write(str(data))
            f.write('\n')

    def extend(self, data: list):
        with open(self.path, 'a') as f:
            f.write('\n'.join(map(str, data)))
            f.write('\n')
