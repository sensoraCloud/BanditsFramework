import pandas as pd
import pytest

from voucher_opt.features.feature_definitions import FeatureType, Feature
from voucher_opt.transformers.categorical_transformer import CategoricalTransformer

CATEGORICAL_FEATURE_NAMES = ['gender', 'city']
CATEGORICAL_FEATURES = [Feature.create(f, FeatureType.CATEGORICAL, None) for f in CATEGORICAL_FEATURE_NAMES]
FEATURES = [Feature.create('age', FeatureType.NUMERICAL, None)] + CATEGORICAL_FEATURES


@pytest.fixture(scope='session')
def train_df():
    train_df = pd.DataFrame(
        [
            [1, 20, 'female', 'New York'],
            [2, 20, 'male', 'Berlin'],
            [3, 40, 'female', 'Tokyo'],
            [4, 70, 'male', 'Berlin'],
            [5, 55, 'female', 'New York']
        ],
        columns=['customer_id'] + [f.short_name for f in FEATURES])
    return train_df


@pytest.fixture(scope='session')
def predict_df():
    predict_df = pd.DataFrame(
        [
            [123, 19, 'female', 'Berlin'],
            [999, 25, 'other', 'Malmö'],
        ],
        columns=['customer_id'] + [f.short_name for f in FEATURES])
    return predict_df


def test_fit_transform(train_df):
    transformer = CategoricalTransformer(CATEGORICAL_FEATURES, 0.25)
    transformed = transformer.fit(train_df)
    expected_columns = ['customer_id', 'a', 'g_female', 'g_male', 'c_Berlin', 'c_New York']
    assert list(transformed.columns) == expected_columns
    assert transformed.shape == (5, 6)
    assert list(transformed[expected_columns[0]].values) == [1, 2, 3, 4, 5]
    assert list(transformed[expected_columns[1]].values) == [20, 20, 40, 70, 55]
    assert list(transformed[expected_columns[2]].values) == [1, 0, 1, 0, 1]
    assert list(transformed[expected_columns[3]].values) == [0, 1, 0, 1, 0]
    assert list(transformed[expected_columns[4]].values) == [0, 1, 0, 1, 0]
    assert list(transformed[expected_columns[5]].values) == [1, 0, 0, 0, 1]


def test_transform(predict_df):
    transformer = CategoricalTransformer(CATEGORICAL_FEATURES, None)
    transformer._feature_columns = ['g_female', 'g_male', 'c_Berlin', 'c_New York', 'c_Tokyo']
    transformed = transformer.transform(predict_df)
    expected_columns = ['customer_id', 'a', 'g_female', 'g_male', 'c_Berlin', 'c_New York', 'c_Tokyo']
    assert list(transformed.columns) == expected_columns
    assert transformed.shape == (2, 7)
    assert list(transformed[expected_columns[0]].values) == [123, 999]
    assert list(transformed[expected_columns[1]].values) == [19, 25]
    assert list(transformed[expected_columns[2]].values) == [1, 0]
    assert list(transformed[expected_columns[3]].values) == [0, 0]
    assert list(transformed[expected_columns[4]].values) == [1, 0]
    assert list(transformed[expected_columns[5]].values) == [0, 0]
    assert list(transformed[expected_columns[6]].values) == [0, 0]
