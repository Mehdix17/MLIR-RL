from rl_autoschedular_v4.state import OperationState
from rl_autoschedular_v4.transforms import transform_pack
from typing import Optional

from utils.config import Config
from .base import Action
import torch
import math
from torch.distributions import Categorical


class Pack(Action):
    """Class representing Pack action"""

    symbol = 'PK'

    parameters: list[int]

    def __init__(self, parameters: list[int], state: Optional[OperationState] = None, **extras):
        if state:
            # Case where state is provided -> Parameters need processing
            pack_sizes = []
            for param, loop in zip(parameters, state.operation_features.nested_loops):
                if param == 0:
                    pack_sizes.append(0)
                else:
                    ps = 2 ** (param - 1)
                    assert loop.upper_bound % ps == 0 and loop.upper_bound != ps, \
                        f'Pack parameter {param} is not a factor of loop upper bound {loop.upper_bound}'
                    pack_sizes.append(ps)
            parameters = pack_sizes
        super().__init__(parameters, state, **extras)

    @classmethod
    def params_size(cls):
        return Config().max_num_loops

    @classmethod
    def network_output_size(cls):
        return Config().max_num_loops * (Config().num_tile_sizes + 1)

    @classmethod
    def history_size(cls):
        return Config().truncate * Config().max_num_loops * (Config().num_tile_sizes + 1)

    @classmethod
    def action_mask(cls, state: OperationState):
        mask = torch.zeros((Config().max_num_loops, Config().num_tile_sizes + 1), dtype=torch.bool)
        mask[:, 0] = True
        for i, loop in enumerate(state.operation_features.nested_loops):
            ps_count = cls.__get_pack_sizes_count(loop.upper_bound)
            mask[i, :ps_count] = True
        return mask.reshape(-1)

    @classmethod
    def action_history(cls, seq):
        history = torch.zeros((Config().truncate, Config().max_num_loops, Config().num_tile_sizes + 1))
        for i, action in enumerate(seq):
            if not isinstance(action, Pack):
                continue
            for j, param in enumerate(action.parameters):
                if param == 0:
                    history[i, j, 0] = 1
                else:
                    assert param > 0 and (param & (param - 1) == 0), f'Expected pack size to be a positive power of 2, found {param}'
                    ps_index = int(math.log2(param)) + 1
                    assert ps_index < history.size(2), f'Overflow of pack size, max size is {2 ** (Config().num_tile_sizes - 1)} found {param}'
                    history[i, j, ps_index] = 1
        return history.reshape(-1)

    @classmethod
    def distribution(cls, logits):
        logits = logits.reshape(-1, Config().max_num_loops, Config().num_tile_sizes + 1)
        return Categorical(logits=logits)

    @classmethod
    def distribution_stats(cls, distribution, index, eps_distribution, eps=None):
        log_p = distribution.log_prob(index).sum(-1)
        if eps is not None:
            eps_log_p = eps_distribution.log_prob(index).sum(-1)
            log_p = (log_p.exp() * (1 - eps) + eps_log_p.exp() * eps).log()
        entropy = distribution.entropy().sum(-1)
        return log_p, entropy

    @classmethod
    def sample(cls, distribution, eps_distribution, num_loops, uniform, greedy):
        if greedy:
            index = distribution.probs.argmax(-1)
        elif uniform:
            index = eps_distribution.sample()
        else:
            index = distribution.sample()
        return index

    def _apply_ready(self, code):
        return transform_pack(code, self.operation_tag, self.parameters)

    def update_features(self, operation_features):
        new_operation_features = operation_features.copy()
        for nested_loop, pack_size in zip(new_operation_features.nested_loops, self.parameters):
            if pack_size == 0:
                continue
            nested_loop.upper_bound = (nested_loop.upper_bound + pack_size - 1) // pack_size
        return new_operation_features

    @staticmethod
    def __get_pack_sizes_count(ub: int) -> int:
        """Get the number of pack size candidates for a given loop upper bound."""
        for i in range(Config().num_tile_sizes):
            ps = 2 ** i
            if ub % ps != 0 or ub == ps:
                return i + 1
        return Config().num_tile_sizes + 1
