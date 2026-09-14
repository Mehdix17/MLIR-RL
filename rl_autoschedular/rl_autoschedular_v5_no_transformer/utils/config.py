from typing import Literal, Any, Optional
from typeguard import check_type, CollectionCheckStrategy
from rl_autoschedular_v5_no_transformer.utils.singleton import Singleton
import json
import os


class Config(metaclass=Singleton):
    """Class to store and load global configuration"""

    max_num_stores_loads: int
    """The maximum number of loads in the nested loops"""
    max_num_loops: int
    """The max number of nested loops"""
    max_num_load_store_dim: int
    """The max number of dimensions in load/store buffers"""
    num_tile_sizes: int
    """The number of tile sizes"""
    num_pad_multiples: int = 3
    """The number of pad multiple candidates (powers of 2: 2, 4, 8)."""
    num_unroll_factors: int = 3
    """The number of unroll factor candidates (powers of 2: 2, 4, 8)."""
    dask_worker_cores: Optional[int] = None
    """Distributed (V5.1): cores per dask worker (env DASK_WORKER_CORES overrides)."""
    dask_worker_mem: Optional[str] = None
    """Distributed (V5.1): memory per dask worker, e.g. '2GB' (env DASK_WORKER_MEM overrides)."""
    dask_worker_exclusive: Optional[bool] = None
    """Distributed (V5.1): node exclusivity per worker (env DASK_WORKER_EXCLUSIVE overrides)."""
    dask_node_count: Optional[int] = None
    """Distributed (V5.1): number of dask workers; 0/absent = single-node mode
    (env DASK_NODES overrides — launch-time smoke tests)."""
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
    normalize_bounds: Literal['none', 'max', 'log']
    """Flag to indicate if the upper bounds in the input should be normalized or not"""
    normalize_adv: Literal['none', 'standard', 'max-abs']
    """The advantage normalization method"""
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
    ppo_epochs: int
    """Number of epochs for PPO"""
    ppo_batch_size: Optional[int]
    """Batch size for PPO"""
    value_epochs: int
    """Number of epochs for value update"""
    value_batch_size: Optional[int]
    """Batch size for value update"""
    value_coef: float
    """Value coefficient"""
    value_clip: bool
    """Clip value loss or not"""
    entropy_coef: float
    """Entropy coefficient"""
    lr: float
    """Learning rate"""
    truncate: int
    """Maximum number of steps in the schedule"""
    json_file: str = ""
    """Path to the JSON file containing the benchmarks execution times.
    When empty (default), the path is auto-derived from results_dir and
    the current implementation via utils.implementation.get_split_file_path."""
    eval_json_file: str = ""
    """Path to the JSON file containing the benchmarks execution times for evaluation.
    When empty (default), the path is auto-derived from results_dir and
    the current implementation via utils.implementation.get_split_file_path."""
    tags: list[str]
    """List of tags to add to the neptune experiment"""
    debug: bool
    """Flag to enable debug mode"""
    main_exec_data_file: str
    """Path to the file containing the execution data"""
    results_dir: str
    """Path to the results directory"""
    implementation: str = "rl_autoschedular_v0"
    """Autoscheduler package implementation to use (e.g., rl_autoschedular, rl_autoschedular_v1)."""
    transformer_d_model: int = 256
    """Hidden size for transformer token embeddings."""
    transformer_nhead: int = 8
    """Number of attention heads in transformer encoder layers."""
    transformer_num_layers: int = 3
    """Number of transformer encoder layers."""
    transformer_ffn_dim: int = 1024
    """Feed-forward hidden dimension used in transformer encoder layers."""
    transformer_dropout: float = 0.1
    """Dropout ratio used by transformer projections and encoder layers."""
    transformer_activation: Literal['relu', 'gelu'] = 'gelu'
    """Activation function used inside transformer encoder feed-forward blocks."""
    transformer_pooling: Literal['cls', 'mean'] = 'cls'
    """Token pooling strategy for transformer output."""
    transformer_use_action_history_token: bool = False
    """If true, action history is injected as a transformer token instead of post-concatenation."""
    eval_runs: int = 1
    """Number of execution runs per benchmark during evaluation (default 1 = single run)."""
    eval_aggregation: Literal['min', 'median', 'mean'] = 'min'
    """Aggregation method for multiple eval runs: min, median, or mean."""
    ppo_clip_range: float = 0.2
    """PPO policy ratio clipping bound."""
    gae_lambda: float = 0.95
    """GAE (Generalized Advantage Estimation) lambda discount factor."""
    max_grad_norm: float = 0.5
    """Maximum gradient norm for clipping."""

    def __init__(self):
        """Load the configuration from the JSON file
        or get existing instance if any.
        """
        # Open the JSON file
        with open(os.getenv("CONFIG_FILE_PATH"), "r") as f:
            config_data: dict[str, Any] = json.load(f)

        for element, element_t in self.__annotations__.items():
            if element in config_data:
                raw_value = config_data[element]
            elif hasattr(self.__class__, element):
                raw_value = getattr(self.__class__, element)
            else:
                raise ValueError(f"{element} is missing from the config file")

            element_v = check_type(raw_value, element_t, collection_check_strategy=CollectionCheckStrategy.ALL_ITEMS)
            setattr(self, element, element_v)

    def to_dict(self):
        """Convert the configuration to a dictionary."""
        return {k: self.__dict__[k] for k in self.__annotations__}

    def __str__(self):
        """Convert the configuration to a string."""
        return str(self.to_dict())
