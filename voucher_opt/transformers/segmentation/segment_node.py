from typing import List, Tuple

import numpy as np
import pandas as pd

from voucher_opt.constants import UNIVERSE, ACTION_CODE
from voucher_opt.logger import log
from voucher_opt.transformers.segmentation.errors import NoSplitCandidatesException
from voucher_opt.transformers.segmentation.tree_segment_parameters import TreeSegmentParameters

FEATURE = 'feature'
FEATURE_VALUE = 'feature_value'
FEATURE_GAIN = 'feature_gain'
FEAT_VAL_ACTION_COUNT = 'feat_val_action_count'
FEAT_VAL_ACTION_REWARD_AGG = 'feat_val_action_reward_agg'
FEAT_VAL_COUNT = 'feat_val_count'
FEAT_VAL_REWARD = 'feat_val_reward'
FEAT_VAL_EXPECTED = 'feat_val_expected'
CHERNOFF_CONFIDENT = 'chernoff_confident'
NORM_FEEDBACK = 'norm_feedback'


class SegmentNode:
    def __init__(self, feature: str, feature_value: int, parent_depth: int, ancestors: List[Tuple],
                 segment_df: pd.DataFrame, parameters: TreeSegmentParameters):
        self._df = segment_df
        self._feature = feature
        self._feature_value = feature_value
        self.left: SegmentNode = None
        self.right: SegmentNode = None
        self.branch = list(ancestors)
        if feature:
            self.branch.append((feature, feature_value))
        self.depth = parent_depth + 1
        self._parameters: TreeSegmentParameters = parameters

    def __str__(self):
        if self._feature is None and self._feature_value is None:
            return UNIVERSE
        return f'{self._feature}:{self._feature_value}'

    def __repr__(self):
        return str(self)

    @property
    def has_features(self):
        return len(self._df.columns) > 2

    def select_split_feature(self, total_num_samples, total_num_features):
        segment_num_samples = len(self._df)
        features_df = self._calculate_feature_gains(segment_num_samples)
        features_df[CHERNOFF_CONFIDENT] = self._is_chernoff_confident(features_df, segment_num_samples,
                                                                      total_num_samples, total_num_features)
        features_df['candidate_split'] = features_df.chernoff_confident & features_df.sample_confident

        log.debug('Split feature candidates:')
        log.debug(list(features_df[features_df.candidate_split][FEATURE]))

        split_candidates = features_df[features_df.candidate_split]
        if split_candidates.empty:
            raise NoSplitCandidatesException
        selected_feature_idx = np.argmax(np.random.multinomial(1, _softmax(split_candidates[FEATURE_GAIN].values)))
        selected_feature = split_candidates[FEATURE].values[selected_feature_idx]

        return selected_feature

    def split_segment(self, split_feature):
        values = self._df[split_feature].unique()
        left_segment_df = self._df[self._df[split_feature] == values[0]].drop(split_feature, axis=1)
        right_segment_df = self._df[self._df[split_feature] == values[1]].drop(split_feature, axis=1)
        self.left = SegmentNode(split_feature, values[0], self.depth, self.branch, left_segment_df,
                                self._parameters)
        self.right = SegmentNode(split_feature, values[1], self.depth, self.branch, right_segment_df,
                                 self._parameters)
        return self.left, self.right

    # TODO: Add docstring (at least input -> output with types and cols specified)
    def _calculate_feature_gains(self, segment_num_samples):
        segment_expected_feedback = self._df[[ACTION_CODE, NORM_FEEDBACK]].groupby([ACTION_CODE]).agg(
            self._parameters.reward_agg_func).max()[0]

        non_feature_cols = [ACTION_CODE, NORM_FEEDBACK]
        split_feature_df = self._df.copy().set_index(non_feature_cols).stack().reset_index()
        split_feature_df.columns = non_feature_cols + [FEATURE, FEATURE_VALUE]

        # Calculate expected feedback for each (feature, value, action)
        split_feature_df = split_feature_df[[FEATURE, FEATURE_VALUE, 'action_code', NORM_FEEDBACK]] \
            .groupby([FEATURE, FEATURE_VALUE, 'action_code']) \
            .agg(['count', self._parameters.reward_agg_func]).reset_index()
        split_feature_df.columns = [FEATURE, FEATURE_VALUE, 'action_code', FEAT_VAL_ACTION_COUNT,
                                    FEAT_VAL_ACTION_REWARD_AGG]

        # Calculate maximum feedback for each (feature, value)
        split_feature_df = split_feature_df \
            .groupby([FEATURE, FEATURE_VALUE]) \
            .apply(feature_value_aggregations, len(self._parameters.actions),
                   self._parameters.min_sample_conf).reset_index()

        # Calculate expected feedback for each (feature, value)
        feat_val_count = split_feature_df[FEAT_VAL_COUNT]
        split_feature_df[FEAT_VAL_EXPECTED] = (feat_val_count / segment_num_samples) * \
                                              split_feature_df[FEAT_VAL_REWARD]

        # Calculate expected feedback for each feature
        split_feature_df = split_feature_df[[FEATURE, FEATURE_VALUE, FEAT_VAL_EXPECTED, 'sample_confident']] \
            .groupby([FEATURE]) \
            .apply(feature_aggregations).reset_index()
        split_feature_df[FEATURE_GAIN] = split_feature_df.feat_expected_feedback - segment_expected_feedback

        return split_feature_df

    def _is_chernoff_confident(self, split_feature_data, segment_num_samples, total_num_samples, total_num_features):
        confidence_bound = self._parameters.tree_conf_bound * \
                           np.sqrt((1 / segment_num_samples) *
                                   np.log(np.power(segment_num_samples, 2) *
                                          np.power(self.depth, 2) *
                                          total_num_samples * total_num_features))
        return split_feature_data[FEATURE_GAIN] >= confidence_bound


def feature_value_aggregations(group, num_actions, tree_min_sample_conf):
    d = {
        FEAT_VAL_COUNT: group[FEAT_VAL_ACTION_COUNT].sum(),
        FEAT_VAL_REWARD: group[FEAT_VAL_ACTION_REWARD_AGG].max()
    }
    confident_actions = np.sum(group[FEAT_VAL_ACTION_COUNT] >= tree_min_sample_conf)
    d['sample_confident'] = confident_actions == num_actions
    return pd.Series(d, index=[FEAT_VAL_COUNT, FEAT_VAL_REWARD, 'sample_confident'])


def feature_aggregations(group):
    d = {'feat_expected_feedback': group[FEAT_VAL_EXPECTED].sum()}
    confident_feature_values = np.sum(group['sample_confident'] == 1)
    d['sample_confident'] = np.sum(confident_feature_values == len(group['sample_confident']))
    return pd.Series(d, index=['feat_expected_feedback', 'sample_confident'])


def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
