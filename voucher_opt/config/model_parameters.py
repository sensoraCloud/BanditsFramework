import numpy as np

from voucher_opt import constants
from voucher_opt.utils import validate_bounds, validate_type


class InvalidConfigError(BaseException):
    pass


class ModelParameters:
    def __init__(self, epsilon, action_confidence_threshold, tree_min_sample_conf, segment_num_trees, tree_max_depth,
                 tree_conf_bound, number_of_bins, categorical_frequency_threshold, reward_agg_function):
        self.epsilon = epsilon
        self.action_confidence_threshold = action_confidence_threshold
        self.tree_min_sample_conf = tree_min_sample_conf
        self.segment_num_trees = segment_num_trees
        self.tree_max_depth = tree_max_depth
        self.tree_conf_bound = tree_conf_bound
        self.number_of_bins = number_of_bins
        self.categorical_frequency_threshold = categorical_frequency_threshold
        self._reward_agg_function = reward_agg_function

    def __str__(self):
        param_strings = [f'epsilon =  {self.epsilon}',
                         f'action_confidence_threshold = {self.action_confidence_threshold}',
                         f'tree_min_sample_conf = {self.tree_min_sample_conf}',
                         f'segment_num_trees = {self.segment_num_trees}',
                         f'tree_max_depth = {self.tree_max_depth}',
                         f'tree_conf_bound = {self.tree_conf_bound}',
                         f'number_of_bins = {self.number_of_bins}',
                         f'categorical_frequency_threshold = {self.categorical_frequency_threshold}',
                         f'reward_agg_function = {self._reward_agg_function}']
        return '\n\t'.join(param_strings)

    @staticmethod
    def create_from_run_config(run_config):
        parameters = ModelParameters(
            run_config[constants.EPSILON_KEY],
            run_config[constants.ACTION_CONFIDENCE_THRESHOLD_KEY],
            run_config[constants.TREE_MIN_SAMPLE_CONF_KEY],
            run_config[constants.SEGMENT_NUM_TREES_KEY],
            run_config[constants.TREE_MAX_DEPTH_KEY],
            run_config[constants.TREE_CONF_BOUND_KEY],
            run_config[constants.NUMBER_OF_BINS_KEY],
            run_config[constants.CATEGORICAL_FREQUENCY_THRESHOLD_KEY],
            run_config[constants.REWARD_AGG_FUNC_KEY]
        )
        parameters.validate()
        return parameters

    def set_from_dict(self, parameters):
        for key in parameters:
            setattr(self, key, parameters[key])

    @property
    def reward_agg_function(self):
        if self._reward_agg_function == 'mean':
            return np.mean
        if self._reward_agg_function == 'median':
            return np.median
        raise InvalidConfigError(
            f'Reward aggregation function "{self._reward_agg_function}" does not exist.')

    def validate(self):
        validate_bounds('action_confidence_threshold', self.action_confidence_threshold, lower_bound=0)
        validate_type('action_confidence_threshold', self.action_confidence_threshold, int)
