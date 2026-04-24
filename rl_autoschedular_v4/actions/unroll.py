from rl_autoschedular_v4.state import OperationState
from rl_autoschedular_v4.transforms import transform_unroll
from typing import Optional

from utils.config import Config
from .base import Action
import torch
import math
from torch.distributions import Categorical


class Unroll(Action):
    """Class representing Unroll action.

    Tiles the operation using tile_using_for with tile sizes equal to
    loop_bound // factor, then unrolls each resulting loop. Because
    unrolling duplicates the tagged op, subsequent tag-based transforms
    may match multiple ops, so this action is terminal.
    """

    symbol = 'U'

    parameters: list[int]

    # --- constants ---
    terminal = True

    def __init__(self, parameters: list[int], state: Optional[OperationState] = None, **extras):
        if state:
            assert len(parameters) == 1, 'Unroll expects exactly one parameter'
            param = parameters[0]
            factor = 2 ** (param + 1)
            tile_sizes = []
            for loop in state.operation_features.nested_loops:
                if loop.upper_bound > 1:
                    assert loop.upper_bound % factor == 0, \
                        f'Unroll factor {factor} does not divide loop bound {loop.upper_bound}'
                    tile_sizes.append(loop.upper_bound // factor)
                else:
                    tile_sizes.append(0)
            extras['tile_sizes'] = tile_sizes
            parameters = [factor]
        super().__init__(parameters, state, **extras)

    @classmethod
    def params_size(cls):
        return 1

    @classmethod
    def network_output_size(cls):
        return Config().num_unroll_factors

    @classmethod
    def history_size(cls):
        return Config().truncate * Config().num_unroll_factors

    @classmethod
    def is_allowed(cls, state: OperationState):
        # Terminal: only allow if there are no producers and there are loops to unroll
        return state.producer_tag is None and any(loop.upper_bound > 1 for loop in state.operation_features.nested_loops)

    @classmethod
    def action_mask(cls, state: OperationState):
        mask = torch.zeros(Config().num_unroll_factors, dtype=torch.bool)
        for i in range(Config().num_unroll_factors):
            factor = 2 ** (i + 1)
            if all(loop.upper_bound % factor == 0 and loop.upper_bound >= factor
                   for loop in state.operation_features.nested_loops if loop.upper_bound > 1):
                mask[i] = True
        return mask

    @classmethod
    def action_history(cls, seq):
        history = torch.zeros((Config().truncate, Config().num_unroll_factors))
        for i, action in enumerate(seq):
            if not isinstance(action, Unroll):
                continue
            param = action.parameters[0]
            assert param > 0 and (param & (param - 1) == 0), f'Expected unroll factor to be a positive power of 2, found {param}'
            uf_index = int(math.log2(param)) - 1
            assert 0 <= uf_index < history.size(1), f'Unroll factor {param} out of range'
            history[i, uf_index] = 1
        return history.reshape(-1)

    @classmethod
    def distribution(cls, logits):
        return Categorical(logits=logits)

    @classmethod
    def distribution_stats(cls, distribution, index, eps_distribution, eps=None):
        index = index.squeeze(-1)
        log_p = distribution.log_prob(index)
        if eps is not None:
            eps_log_p = eps_distribution.log_prob(index)
            log_p = (log_p.exp() * (1 - eps) + eps_log_p.exp() * eps).log()
        entropy = distribution.entropy()
        return log_p, entropy

    @classmethod
    def sample(cls, distribution, eps_distribution, num_loops, uniform, greedy):
        if greedy:
            index = distribution.probs.argmax(-1)
        elif uniform:
            index = eps_distribution.sample()
        else:
            index = distribution.sample()
        return index.unsqueeze(-1)

    def _apply_ready(self, code):
        return transform_unroll(code, self.operation_tag, self.extras['tile_sizes'], self.parameters[0])

    def update_features(self, operation_features):
        new_operation_features = operation_features.copy()
        for nested_loop, tile_size in zip(new_operation_features.nested_loops, self.extras['tile_sizes']):
            if tile_size == 0:
                continue
            nested_loop.upper_bound = tile_size
        return new_operation_features
