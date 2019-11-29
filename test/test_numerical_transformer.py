import numpy as np
import pandas as pd
import pytest

from voucher_opt.features.feature_definitions import FeatureType, Feature
from voucher_opt.transformers.numerical_transformer import NumericalFeatureTransformer

NUMERICAL_FEATURE_NAMES = ['age', 'height', 'weight', 'shoe_size']
NUMERICAL_FEATURES = [Feature.create(f, FeatureType.NUMERICAL, None) for f in NUMERICAL_FEATURE_NAMES]
FEATURES = [Feature.create('gender', FeatureType.CATEGORICAL, None)] + NUMERICAL_FEATURES


# TODO: Add tests for both int and float features
# TODO: Test for different values of PERCENTILE_BINNING_STEP

@pytest.fixture(scope='session')
def train_df():
    train_df = pd.DataFrame(
        [
            [1, 'female', 10, 120, 40, 35],
            [2, 'male', 20, 170, 60, 45],
            [3, 'female', 40, 150, 90, 39],
            [4, 'male', 70, 210, 70, 52]
        ],
        columns=['customer_id'] + [f.short_name for f in FEATURES])
    return train_df


@pytest.fixture(scope='session')
def predict_df():
    predict_df = pd.DataFrame(
        [
            [123, 'female', 19, 148, 95, 36],
            [999, 'other', 25, 195, 80, 42],
        ],
        columns=['customer_id'] + [f.short_name for f in FEATURES])
    return predict_df


def test_fit_transform(train_df):
    transformer = NumericalFeatureTransformer(NUMERICAL_FEATURES, 4)
    transformed = transformer.fit(train_df)
    expected_columns = ['customer_id', 'g', 'b_a_0', 'b_a_1', 'b_a_2', 'b_a_3', 'b_h_0', 'b_h_1', 'b_h_2', 'b_h_3',
                        'b_w_0', 'b_w_1', 'b_w_2', 'b_w_3', 'b_ss_0', 'b_ss_1', 'b_ss_2', 'b_ss_3']
    assert list(transformed.columns) == expected_columns
    assert transformed.shape == (4, 18)
    assert list(transformed[expected_columns[0]].values) == [1, 2, 3, 4]
    assert list(transformed[expected_columns[1]].values) == ['female', 'male', 'female', 'male']
    assert list(transformed[expected_columns[2]].values) == [1, 0, 0, 0]
    assert list(transformed[expected_columns[3]].values) == [0, 1, 0, 0]
    assert list(transformed[expected_columns[4]].values) == [0, 0, 1, 0]
    assert list(transformed[expected_columns[5]].values) == [0, 0, 0, 1]
    assert list(transformed[expected_columns[6]].values) == [1, 0, 0, 0]
    assert list(transformed[expected_columns[7]].values) == [0, 0, 1, 0]
    assert list(transformed[expected_columns[8]].values) == [0, 1, 0, 0]
    assert list(transformed[expected_columns[9]].values) == [0, 0, 0, 1]
    assert list(transformed[expected_columns[10]].values) == [1, 0, 0, 0]
    assert list(transformed[expected_columns[11]].values) == [0, 1, 0, 0]
    assert list(transformed[expected_columns[12]].values) == [0, 0, 0, 1]
    assert list(transformed[expected_columns[13]].values) == [0, 0, 1, 0]
    assert list(transformed[expected_columns[14]].values) == [1, 0, 0, 0]
    assert list(transformed[expected_columns[15]].values) == [0, 0, 1, 0]
    assert list(transformed[expected_columns[16]].values) == [0, 1, 0, 0]
    assert list(transformed[expected_columns[17]].values) == [0, 0, 0, 1]


def test_transform(predict_df):
    transformer = NumericalFeatureTransformer(NUMERICAL_FEATURES, None)
    transformer._feature_to_bins = {'a': np.array([1, 20, 30]),
                                    'h': np.array([120, 150, 190]),
                                    'w': np.array([90, 150]),
                                    'ss': np.array([30, 35, 40, 45])
                                    }
    transformed = transformer.transform(predict_df)
    expected_columns = ['customer_id', 'g', 'b_a_0', 'b_a_1', 'b_h_0', 'b_h_1', 'b_w_0', 'b_ss_0', 'b_ss_1', 'b_ss_2']
    assert list(transformed.columns) == expected_columns
    assert list(transformed[expected_columns[0]].values) == [123, 999]
    assert list(transformed[expected_columns[1]].values) == ['female', 'other']
    assert list(transformed[expected_columns[2]].values) == [1, 0]
    assert list(transformed[expected_columns[3]].values) == [0, 1]
    assert list(transformed[expected_columns[4]].values) == [1, 0]
    assert list(transformed[expected_columns[5]].values) == [0, 0]
    assert list(transformed[expected_columns[6]].values) == [1, 0]
    assert list(transformed[expected_columns[7]].values) == [0, 0]
    assert list(transformed[expected_columns[8]].values) == [1, 0]
    assert list(transformed[expected_columns[9]].values) == [0, 1]


def test_fit_and_transform(train_df, predict_df):
    transformer = NumericalFeatureTransformer(NUMERICAL_FEATURES, 4)
    train_transformed = transformer.fit(train_df)
    expected_columns = ['customer_id', 'g', 'b_a_0', 'b_a_1', 'b_a_2', 'b_a_3', 'b_h_0', 'b_h_1', 'b_h_2', 'b_h_3',
                        'b_w_0', 'b_w_1', 'b_w_2', 'b_w_3', 'b_ss_0', 'b_ss_1', 'b_ss_2', 'b_ss_3']
    assert list(train_transformed.columns) == expected_columns
    assert train_transformed.shape == (4, 18)

    predict_transformed = transformer.transform(predict_df)
    assert list(predict_transformed[expected_columns[0]].values) == [123, 999]
    assert list(predict_transformed[expected_columns[1]].values) == ['female', 'other']
    assert list(predict_transformed[expected_columns[2]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[3]].values) == [1, 1]
    assert list(predict_transformed[expected_columns[4]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[5]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[6]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[7]].values) == [1, 0]
    assert list(predict_transformed[expected_columns[8]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[9]].values) == [0, 1]
    assert list(predict_transformed[expected_columns[10]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[11]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[12]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[13]].values) == [0, 1]
    assert list(predict_transformed[expected_columns[14]].values) == [1, 0]
    assert list(predict_transformed[expected_columns[15]].values) == [0, 1]
    assert list(predict_transformed[expected_columns[16]].values) == [0, 0]
    assert list(predict_transformed[expected_columns[17]].values) == [0, 0]
