import operator
from collections import deque
from functools import reduce
from typing import List, Tuple, Deque

import numpy as np
import pandas as pd

from voucher_opt.config.model_parameters import ModelParameters
from voucher_opt.constants import SEGMENT, UNIVERSE, GIVER_ID, HAS_EXPERIAN
from voucher_opt.logger import log
from voucher_opt.transformers.segmentation.errors import NoSplitCandidatesException
from voucher_opt.transformers.segmentation.segment_node import SegmentNode
from voucher_opt.transformers.segmentation.tree_segment_parameters import TreeSegmentParameters
from voucher_opt.transformers.transformer import Transformer
from voucher_opt.utils import get_feature_columns


class TreeSegmentTransformer(Transformer):
    def __init__(self, non_feature_columns, actions, has_experian, model_parameters: ModelParameters):
        self._segmentation: List[List[Tuple[str, int]]] = [[()]]
        self._non_feature_columns = non_feature_columns
        self._parameters = TreeSegmentParameters(actions, model_parameters.tree_min_sample_conf,
                                                 model_parameters.segment_num_trees, model_parameters.tree_max_depth,
                                                 model_parameters.tree_conf_bound, model_parameters.reward_agg_function)
        self._input_feature_columns = None
        self._has_experian = has_experian

    def fit(self, df: pd.DataFrame):
        original_df = df.copy()
        df = df.drop([GIVER_ID], axis=1)
        self._input_feature_columns = get_feature_columns(df, self._non_feature_columns)
        self._segmentation = self._random_forest_segmentation(df)
        if not _valid_segmentation(self._segmentation):
            log.info('Could not find a confident segmentation.')
        df = self._segment_dataframe(original_df, self._segmentation)
        log.info(f'Final segmentation: {", ".join(df.segment.unique())}')
        return df

    def transform(self, df: pd.DataFrame):
        if not _valid_segmentation(self._segmentation):
            df = set_universe_segment(df)
            return df
        assert sorted(get_feature_columns(df, self._non_feature_columns)) == sorted(self._input_feature_columns)
        return self._segment_dataframe(df, self._segmentation)

    def _random_forest_segmentation(self, dataset) -> List[List[Tuple[str, int]]]:
        log.debug('Dataset before running tree algorithm:\n'
                  f'{dataset.head().to_string()}\n')

        max_gain = 0.0
        max_segmentation: List[List[Tuple[str, int]]] = []
        log.info('Creating segmentation...')
        for _ in range(self._parameters.segment_num_trees):

            dataset.loc[:, 'norm_feedback'] = min_max_normalize(dataset['feedback'])

            segmentation_builder = TreeSegmentationBuilder(dataset.drop('feedback', axis=1), self._parameters)
            tree = segmentation_builder.build_tree(self._has_experian)
            segments = _get_segments(tree)
            segmentation_df = self._segment_dataframe(dataset, segments)
            tree_gain_vs_no_action = self._evaluate_segmentation(segmentation_df)

            if tree_gain_vs_no_action > max_gain:
                max_gain = tree_gain_vs_no_action
                max_segmentation = segments

        log.debug(f'Expected gain of the best tree vs no action = {max_gain}\n')
        log.info(f'Segments: {max_segmentation}')

        return max_segmentation

    def _segment_dataframe(self, df, segments):
        if _valid_segmentation(segments):
            df = pd.concat([self._select_segment(df, segment) for segment in segments], sort=True)
        else:
            df = set_universe_segment(df)
        return df

    @staticmethod
    def _select_segment(df, segment):
        segment_df = df[reduce(operator.and_, [df[feat_val[0]] == feat_val[1] for feat_val in segment])].copy()
        if segment_df.empty:
            return pd.DataFrame(columns=segment_df.columns)
        segment_df.loc[:, SEGMENT] = '-'.join([f'{feat_val[0]}:{feat_val[1]}' for feat_val in segment])
        return segment_df

    def _evaluate_segmentation(self, segmentation_df):
        max_feedback_per_segment = segmentation_df[['segment', 'action_code', 'feedback']] \
            .groupby(['segment', 'action_code']).agg(self._parameters.reward_agg_func) \
            .reset_index()[['segment', 'feedback']] \
            .groupby('segment').max()

        num_samples_per_segment = segmentation_df[['segment', 'feedback']] \
            .groupby(['segment']).count() \
            .rename(columns={'feedback': 'num_samples'})

        total_num_samples = np.sum(num_samples_per_segment.values)
        expected_value = np.dot(max_feedback_per_segment.values.transpose(),
                                num_samples_per_segment.values / total_num_samples)[0][0]
        log.debug(f'Expected value of segmentation = {expected_value}')

        # Expected value of zero action
        all_zero_exp = segmentation_df[segmentation_df.action_code == 0.0]['feedback'].agg(
            self._parameters.reward_agg_func)

        log.debug(f'Expected value of zero action = {all_zero_exp}')

        tree_gain_vs_no_action = expected_value - all_zero_exp
        log.debug(f'Expected gain of the tree compared to zero action = {tree_gain_vs_no_action}')
        return tree_gain_vs_no_action


def _valid_segmentation(segments):
    return len(segments) > 0 and _no_empty_segments(segments) and _all_segments_valid(segments)


def _no_empty_segments(segments):
    return all([len(segment) > 0 for segment in segments])


def _all_segments_valid(segments):
    return all([len(feat_tup) == 2 for segment in segments for feat_tup in segment])


class TreeSegmentationBuilder:
    def __init__(self, universe_df, parameters: TreeSegmentParameters):
        self._universe_df = universe_df
        self._total_num_samples = len(self._universe_df)
        self._total_num_features = len(self._universe_df.columns.drop(['action_code', 'norm_feedback']))
        self._parameters = parameters
        self._leaves_to_explore = deque()

    def build_tree(self, has_experian=False) -> SegmentNode:
        root = SegmentNode(None, None, 0, [], self._universe_df, self._parameters)
        self._leaves_to_explore.append(root)
        while self._leaves_to_explore:
            log.debug(f'Number of segments so far:')
            log.debug(len(_get_segments(root)))

            log.debug(f'Number of unfinished segments:')
            log.debug(len(self._leaves_to_explore))
            log.debug(f'Unfinished segments: { self._leaves_to_explore }')

            segment_node: SegmentNode = self._leaves_to_explore.popleft()

            log.debug(f'Trying to split segment {segment_node}...')

            if has_experian and segment_node.depth == 1:
                split_feature = HAS_EXPERIAN
            else:
                try:
                    split_feature = segment_node.select_split_feature(self._total_num_samples, self._total_num_features)
                except NoSplitCandidatesException:
                    log.debug(f'No more confident _features for segment {segment_node}')
                    continue

            log.debug(f'Splitting segment {segment_node} on feature {split_feature}...')
            left, right = segment_node.split_segment(split_feature)
            if left.depth < self._parameters.tree_max_depth and left.has_features and right.has_features:
                self._leaves_to_explore.extend([left, right])

        log.debug('No more confident features for any of the segments.')
        log.debug('Segmentation done!')

        return root


def _get_segments(root) -> List[List[Tuple[str, int]]]:
    segments: List[List[Tuple[str, int]]] = []
    q: Deque[SegmentNode] = deque([root])
    while q:
        node: SegmentNode = q.popleft()
        if node.left is None and node.right is None:
            segments.append(node.branch)
            continue
        q.append(node.left)
        q.append(node.right)
    return segments


def set_universe_segment(df):
    df.loc[:, SEGMENT] = UNIVERSE
    return df


def min_max_normalize(df_col):
    col_min = df_col.min()
    return (df_col - col_min) / (df_col.max() - col_min)
