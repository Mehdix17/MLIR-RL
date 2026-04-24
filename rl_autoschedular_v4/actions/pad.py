from rl_autoschedular_v4.state import OperationState
from rl_autoschedular_v4.transforms import transform_pad
from typing import Optional

from utils.config import Config
from .base import Action
import torch
import math
from torch.distributions import Categorical


class Pad(Action):
    """Class representing Pad action"""

    symbol = 'P'

    parameters: list[int]

    def __init__(self, parameters: list[int], state: Optional[OperationState] = None, **extras):
        if state:
            # Parameters need processing: convert index to multiple
            pad_multiples = []
            for param, loop in zip(parameters, state.operation_features.nested_loops):
                if param == 0:
                    pad_multiples.append(0)
                else:
                    multiple = 2 ** param
                    assert multiple > 0, f'Pad parameter {param} is invalid'
                    pad_multiples.append(multiple)
            parameters = pad_multiples
        super().__init__(parameters, state, **extras)

    @classmethod
    def params_size(cls):
        return Config().max_num_loops

    @classmethod
    def network_output_size(cls):
        return Config().max_num_loops * (Config().num_pad_multiples + 1)

    @classmethod
    def history_size(cls):
        return Config().truncate * Config().max_num_loops * (Config().num_pad_multiples + 1)

    @classmethod
    def action_mask(cls, state: OperationState):
        mask = torch.zeros((Config().max_num_loops, Config().num_pad_multiples + 1), dtype=torch.bool)
        mask[:, 0] = True
        for i, loop in enumerate(state.operation_features.nested_loops):
            mult_count = cls.__get_pad_multiples_count(loop.upper_bound)
            mask[i, :mult_count] = True
        return mask.reshape(-1)

    @classmethod
    def action_history(cls, seq):
        history = torch.zeros((Config().truncate, Config().max_num_loops, Config().num_pad_multiples + 1))
        for i, action in enumerate(seq):
            if not isinstance(action, Pad):
                continue
            for j, param in enumerate(action.parameters):
                if param == 0:
                    history[i, j, 0] = 1
                else:
                    assert param > 0 and (param & (param - 1) == 0), f'Expected pad multiple to be a positive power of 2, found {param}'
                    pm_index = int(math.log2(param))
                    assert pm_index < history.size(2), f'Overflow of pad multiple, max is {2 ** (Config().num_pad_multiples - 1)} found {param}'
                    history[i, j, pm_index] = 1
        return history.reshape(-1)

    @classmethod
    def distribution(cls, logits):
        logits = logits.reshape(-1, Config().max_num_loops, Config().num_pad_multiples + 1)
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
        padding_dimensions = []
        pad_to_multiple_of = []
        for dim_idx, multiple in enumerate(self.parameters):
            if multiple == 0:
                continue
            padding_dimensions.append(dim_idx)
            pad_to_multiple_of.append(multiple)
        return transform_pad(code, self.operation_tag, padding_dimensions, pad_to_multiple_of)

    def update_features(self, operation_features):
        new_operation_features = operation_features.copy()
        for nested_loop, multiple in zip(new_operation_features.nested_loops, self.parameters):
            if multiple == 0:
                continue
            # Round up to the nearest multiple
            nested_loop.upper_bound = ((nested_loop.upper_bound + multiple - 1) // multiple) * multiple
        return new_operation_features

    @staticmethod
    def __get_pad_multiples_count(ub: int) -> int:
        """Get the number of pad multiple candidates for a given loop upper bound."""
        for i in range(Config().num_pad_multiples):
            pm = 2 ** (i + 1)
            if pm >= ub:
                return i + 1
        return Config().num_pad_multiples + 1
