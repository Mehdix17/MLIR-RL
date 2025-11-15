from typing import Literal, Any, Optional
from typeguard import check_type, CollectionCheckStrategy
from .singleton import Singleton
import json
import os


class Config(metaclass=Singleton):
    """Class to store and load global configuration
    
    Supports both flat and nested JSON structures:
    - Flat: All parameters at root level (legacy)
    - Nested: Organized into sections (observation_space, action_space, ppo, etc.)
    """

    model_type: Literal['lstm', 'distilbert', 'bert', 'convnext']
    """The type of neural network model to use for policy and value networks"""
    model_config: dict[str, Any]
    """Model-specific configuration (empty for LSTM, populated for transformers)"""
    max_num_stores_loads: int
    """The maximum number of loads in the nested loops"""
    max_num_loops: int
    """The max number of nested loops"""
    max_num_load_store_dim: int
    """The max number of dimensions in load/store buffers"""
    num_tile_sizes: int
    """The number of tile sizes"""
    vect_size_limit: int
    """Vectorization size limit to prevent large sizes vectorization"""
    order: list[list[str]]
    """The order of actions that needs to bo followed"""
    interchange_mode: Literal['enumerate', 'pointers', 'continuous']
    """The method used for interchange action"""
    exploration: list[Literal['entropy', 'epsilon']]
    """The exploration method"""
    init_epsilon: float
    """The initial epsilon value for epsilon greedy exploration"""
    new_architecture: bool
    """Flag to use new architecture"""
    activation: str
    """Activation function"""
    normalize_bounds: Literal['none', 'max', 'log']
    """Flag to indicate if the upper bounds in the input should be normalized or not"""
    normalize_adv: Literal['none', 'standard', 'max-abs']
    """The advantage normalization method"""
    sparse_reward: bool
    """Use sparse rewards"""
    split_ops: bool
    """Split operations"""
    reuse_experience: Literal['none', 'random', 'topk']
    """Strategy for experience replay"""
    benchmarks_folder_path: str
    """Path to the benchmarks folder. Can be empty if optimization mode is set to "last"."""
    bench_count: int
    """Number of batches in a trajectory"""
    replay_count: int
    """Number of trajectories to keep in the replay buffer"""
    nb_iterations: int
    """Number of iterations"""
    truncate: int
    """Maximum number of steps in the schedule"""
    ppo_epochs: int
    """Number of epochs for PPO"""
    ppo_batch_size: Optional[int]
    """Batch size for PPO"""
    lr: float
    """Learning rate"""
    gamma: float
    """Discount factor"""
    clip_epsilon: float
    """PPO clipping parameter"""
    entropy_coef: float
    """Entropy coefficient"""
    value_epochs: int
    """Number of epochs for value update"""
    value_batch_size: Optional[int]
    """Batch size for value update"""
    value_coef: float
    """Value coefficient"""
    value_clip: bool
    """Clip value loss or not"""
    json_file: str
    """Path to the JSON file containing the benchmarks execution times."""
    eval_json_file: str
    """Path to the JSON file containing the benchmarks execution times for evaluation."""
    tags: list[str]
    """List of tags to add to the neptune experiment"""
    debug: bool
    """Flag to enable debug mode"""
    main_exec_data_file: str
    """Path to the file containing the execution data"""
    results_dir: str
    """Path to the results directory"""

    def __init__(self):
        """Load the configuration from the JSON file
        or get existing instance if any.
        
        Supports both flat and nested config structures.
        """
        # Open the JSON file
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as f:
            config_data: dict[str, Any] = json.load(f)

        # Flatten nested structure if present
        flattened_config = self._flatten_config(config_data)
        
        # Set default values for optional parameters
        self._set_defaults(flattened_config)
        
        # Validate and set all required parameters
        for element, element_t in self.__annotations__.items():
            if element not in flattened_config:
                raise ValueError(f"{element} is missing from the config file")

            element_v = check_type(
                flattened_config[element], 
                element_t, 
                collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS
            )
            setattr(self, element, element_v)
    
    def _flatten_config(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Flatten nested config structure to flat structure.
        
        Converts new nested format:
        {
            "observation_space": {"max_num_loops": 12, ...},
            "ppo": {"lr": 0.001, ...},
            ...
        }
        
        To flat format:
        {
            "max_num_loops": 12,
            "lr": 0.001,
            ...
        }
        
        If config is already flat, returns it unchanged.
        """
        # Check if config is already flat (has required root-level params)
        if "max_num_loops" in config_data and "ppo_epochs" in config_data:
            # Already flat format - backward compatible
            return config_data
        
        # New nested format - flatten it
        flattened = {}
        
        # Copy top-level simple fields
        for key in ["model_type", "model_config", "_description"]:
            if key in config_data:
                flattened[key] = config_data[key]
        
        # Flatten nested sections
        section_mappings = {
            "observation_space": ["max_num_stores_loads", "max_num_loops", 
                                  "max_num_load_store_dim", "num_tile_sizes", "vect_size_limit"],
            "action_space": ["order", "interchange_mode"],
            "exploration": ["strategy", "init_epsilon"],
            "architecture": ["new_architecture", "activation", "normalize_bounds", "normalize_adv"],
            "reward": ["sparse_reward", "split_ops"],
            "training": ["bench_count", "replay_count", "nb_iterations", "reuse_experience", "truncate"],
            "ppo": ["ppo_epochs", "ppo_batch_size", "lr", "gamma", "clip_epsilon", "entropy_coef"],
            "value_function": ["value_epochs", "value_batch_size", "value_coef", "value_clip"],
            "data_paths": ["benchmarks_folder_path", "json_file", "eval_json_file"],
            "augmentation": ["use_augmentation", "augmentation_ratio", "augmentation_folder_path",
                           "augmentation_json_file", "neural_nets_folder_path"],
            "logging": ["tags", "debug", "results_dir", "main_exec_data_file"]
        }
        
        for section, keys in section_mappings.items():
            if section in config_data:
                section_data = config_data[section]
                for key in keys:
                    if key in section_data:
                        flattened[key] = section_data[key]
        
        # Handle special case: exploration.strategy -> exploration
        if "strategy" in flattened:
            flattened["exploration"] = flattened.pop("strategy")
        
        return flattened
    
    def _set_defaults(self, config: dict[str, Any]):
        """Set default values for optional parameters."""
        defaults = {
            "model_config": {},
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "new_architecture": False,
            "activation": "relu",
            "sparse_reward": True,
            "split_ops": True,
        }
        
        for key, default_value in defaults.items():
            if key not in config:
                config[key] = default_value

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {k: self.__dict__[k] for k in self.__annotations__}

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.to_dict())
