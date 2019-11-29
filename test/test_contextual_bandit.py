import numpy as np
import pandas as pd

from voucher_opt.constants import SEGMENT, GIVER_ID, CONTROL, EXPLORATION, ACTION_CODE, ACTION_REWARD, ACTION_SAMPLES
from voucher_opt.transformers.contextual_bandit import EpsGreedyBandit, ACTION_DISTRIBUTION


def test_fit_transform_1_non_confident_segment():
    training_df = pd.DataFrame([
        [1, 1, 0.0, 500],
        [1, 2, 100.0, 500],
        [1, 3, 500.0, 500],
        [2, 1, 100.0, 500],
        [2, 2, 500.0, 500],
        [2, 3, 200.0, 500],
        [3, 1, 100.0, 500],
        [3, 2, 0.0, 0],
        [3, 3, 500.0, 500],
    ], columns=[SEGMENT, ACTION_CODE, ACTION_REWARD, ACTION_SAMPLES])
    bandit = EpsGreedyBandit(1, 0.1, [1, 2, 3], 500, 1.0)
    transformed = bandit.fit(training_df)
    assert transformed[transformed[SEGMENT] == 1][ACTION_DISTRIBUTION].values[0] == [0.05, 0.05, 0.9]
    assert transformed[transformed[SEGMENT] == 2][ACTION_DISTRIBUTION].values[0] == [0.05, 0.9, 0.05]
    assert transformed[transformed[SEGMENT] == 3][ACTION_DISTRIBUTION].values[0] == [1.0, 0.0, 0.0]


def test_fit_transform_all_confident():
    training_df = pd.DataFrame([
        [1, 1, 0.0, 500],
        [1, 2, 100.0, 500],
        [1, 3, 500.0, 500],
        [2, 1, 100.0, 500],
        [2, 2, 500.0, 500],
        [2, 3, 200.0, 500],
        [3, 1, 100.0, 500],
        [3, 2, 100.0, 500],
        [3, 3, 500.0, 500],
    ], columns=[SEGMENT, ACTION_CODE, ACTION_REWARD, ACTION_SAMPLES])
    bandit = EpsGreedyBandit(1, 0.1, [1, 2, 3], 500, 1.0)
    transformed = bandit.fit(training_df)
    assert transformed[transformed[SEGMENT] == 1][ACTION_DISTRIBUTION].values[0] == [0.05, 0.05, 0.9]
    assert transformed[transformed[SEGMENT] == 2][ACTION_DISTRIBUTION].values[0] == [0.05, 0.9, 0.05]
    assert transformed[transformed[SEGMENT] == 3][ACTION_DISTRIBUTION].values[0] == [0.05, 0.05, 0.9]


def test_transform():
    bandit = EpsGreedyBandit(1, 0.1, [1, 2, 3], 500, 1.0)
    bandit._eps_greedy_distribution = pd.DataFrame([
        [1, [0.0, 0.0, 1.0]],
        [2, [0.0, 1.0, 0.0]],
    ], columns=[SEGMENT, ACTION_DISTRIBUTION])
    predict_df = pd.DataFrame([
        [1, 2, 0.0, 0.0],
        [2, 1, 0.0, 0.0],
        [3, 2, 0.0, 0.0],
    ], columns=[GIVER_ID, SEGMENT, CONTROL, EXPLORATION])
    actions = bandit.transform(predict_df)
    assert np.array_equal(actions[ACTION_CODE].values, np.array([2, 3, 2]))
