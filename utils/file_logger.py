from .singleton import Singleton
from .config import Config
import json
import os


class FileLogger(metaclass=Singleton):
    """Class to log results to files"""
    def __init__(self):
        cfg = Config()
        tags = ['ppo'] + cfg.tags

        # Write directly to results_dir (no run_N/ subdirectory — v4.9-style flat layout)
        self.run_dir = cfg.results_dir
        os.makedirs(self.run_dir, exist_ok=True)

        # Create tags file
        tags_file = os.path.join(self.run_dir, 'tags')
        with open(tags_file, 'w') as f:
            f.write('\n'.join(tags))
            f.write('\n')

        # Create exec data file inside logs/
        self.logs_dir = os.path.join(self.run_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self.exec_data_file = os.path.join(self.logs_dir, 'exec_data.json')
        if not os.path.exists(self.exec_data_file):
            with open(self.exec_data_file, "w") as f:
                json.dump({}, f)

        # Create models dir
        self.models_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Results file for train progress detection
        self.train_results_file = os.path.join(self.run_dir, 'train_results.json')

        # Init files dict
        self.files_dict: dict[str, FileInstance] = {}

    def __getitem__(self, path: str):
        if path not in self.files_dict:
            full_path = os.path.join(self.logs_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            # File might exist if resuming
            self.files_dict[path] = FileInstance(full_path)
        return self.files_dict[path]

    def clear_per_iter_logs(self):
        pass


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
