import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pytest import approx

from voucher_opt.transformers.segmentation.segment_node import feature_value_aggregations, FEAT_VAL_COUNT, FEAT_VAL_REWARD, \
    feature_aggregations, _softmax
from voucher_opt.transformers.segmentation.tree_segment_transformer import SegmentNode, TreeSegmentParameters, \
    min_max_normalize

COLUMNS = ['country', 'assigned_date', 'receiver_id', 'model_id', 'action_code', 'feedback', 'giver_id', 'g_male',
           'g_female', 'b_a_0', 'b_a_1', 'b_h_0', 'b_h_1']


@pytest.fixture(scope='session')
def actions():
    return [1, 2, 3]


def test_feature_value_aggregations_confident(actions):
    group = pd.DataFrame([
        ['has_sent_freebie', 0, 1, 300, 0.05],
        ['has_sent_freebie', 0, 2, 100, 0.02],
        ['has_sent_freebie', 0, 3, 100, 0.10],
    ], columns=['feat', 'feat_value', 'action_code', 'feat_val_action_count', 'feat_val_action_reward_agg'])
    agg = feature_value_aggregations(group, len(actions), 100)
    assert agg['sample_confident']
    assert agg[FEAT_VAL_COUNT] == 500
    assert agg[FEAT_VAL_REWARD] == 0.1


def test_feature_value_aggregations_not_confident(actions):
    group = pd.DataFrame([
        ['has_sent_freebie', 0, 1, 300, 0.05],
        ['has_sent_freebie', 0, 2, 90, 0.02],
        ['has_sent_freebie', 0, 3, 100, 0.10],
    ], columns=['feat', 'feat_value', 'action_code', 'feat_val_action_count', 'feat_val_action_reward_agg'])
    agg = feature_value_aggregations(group, len(actions), 100)
    assert not agg['sample_confident']
    assert agg[FEAT_VAL_COUNT] == 490
    assert agg[FEAT_VAL_REWARD] == 0.1


def test_feature_aggregations_confident():
    group = pd.DataFrame([
        ['has_sent_freebie', 0.1, True],
        ['has_sent_freebie', 0.2, True],
    ], columns=['feat', 'feat_val_expected', 'sample_confident'])
    agg = feature_aggregations(group)
    assert agg['feat_expected_feedback'] == approx(0.3, 0.001)
    assert agg['sample_confident']


def test_feature_aggregations_not_confident():
    group = pd.DataFrame([
        ['has_sent_freebie', 0.4, True],
        ['has_sent_freebie', 0.2, False],
    ], columns=['feat', 'feat_val_expected', 'sample_confident'])
    agg = feature_aggregations(group)
    assert agg['feat_expected_feedback'] == approx(0.6, 0.001)
    assert not agg['sample_confident']


def test_select_split_feature(actions):
    segment_df = pd.DataFrame([
        [1, 0, 1, 0.01],
        [2, 1, 1, 0.01],
        [3, 0, 1, 0.01],
        [1, 1, 0, 0.5],
        [2, 0, 0, 0.5],
        [3, 1, 0, 0.5],
    ], columns=['action_code', 'has_sent_freebie', 'is_linux_user', 'norm_feedback'])
    node = SegmentNode(None, None, 0, [], segment_df, TreeSegmentParameters(actions, 1, 10, 6, 0.1, np.mean))
    selected_feature = node.select_split_feature(len(segment_df), len(segment_df.columns))
    assert selected_feature == 'has_sent_freebie'


def test_split_segment():
    segment_df = pd.DataFrame([
        [1, 0, 1, 0.01],
        [2, 1, 1, 0.01],
        [3, 0, 1, 0.01],
        [1, 1, 0, 0.5],
        [2, 0, 0, 0.5],
        [3, 1, 0, 0.5],
    ], columns=['action_code', 'has_sent_freebie', 'is_linux_user', 'norm_feedback'])
    node = SegmentNode(None, None, 0, [], segment_df, TreeSegmentParameters(actions, 1, 10, 6, 0.1, np.mean))
    left, right = node.split_segment('has_sent_freebie')
    assert len(left._df) == 3
    assert left._feature == 'has_sent_freebie'
    assert left._feature_value == 0
    assert len(right._df) == 3
    assert right._feature == 'has_sent_freebie'
    assert right._feature_value == 1
    assert left.branch == [('has_sent_freebie', 0)]
    assert right.branch == [('has_sent_freebie', 1)]
    assert left.depth == 2
    assert right.depth == 2


def test_soft_max():
    assert_allclose(_softmax([1, 1, 1]), [0.33333333, 0.33333333, 0.33333333])
    assert_allclose(_softmax([4, 1, 1]), [0.909443, 0.0452785, 0.0452785])


def test_min_max_normalize():
    assert (min_max_normalize(pd.Series([1, 2, 3, 4, 5])) == pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])).all()
