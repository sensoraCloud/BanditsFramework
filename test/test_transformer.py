import pandas as pd

from voucher_opt.features.feature_definitions import Feature, FeatureType
from voucher_opt.transformers.transformer import FeatureNameTransformer, ColumnNameTransformer


def test_feature_name_fit_transform():
    features = [Feature.create('boxes_shipped', FeatureType.NUMERICAL, None),
                Feature.create('activation_channel', FeatureType.CATEGORICAL, None)]
    transformer = FeatureNameTransformer(features)
    df = pd.DataFrame(columns=['customer_id', 'country', 'boxes_shipped', 'activation_channel'])
    transformed = transformer.fit(df)
    assert list(transformed.columns) == ['customer_id', 'country', 'bs', 'ac']


def test_feature_name_transform():
    features = [Feature.create('boxes_shipped', FeatureType.NUMERICAL, None),
                Feature.create('activation_channel', FeatureType.CATEGORICAL, None)]
    transformer = FeatureNameTransformer(features)
    df = pd.DataFrame(columns=['customer_id', 'country', 'boxes_shipped', 'activation_channel'])
    transformed = transformer.transform(df)
    assert list(transformed.columns) == ['customer_id', 'country', 'bs', 'ac']


def test_column_name_fit_transform():
    transformer = ColumnNameTransformer()
    df = pd.DataFrame(columns=['ac_Refer A Friend', 'ac_HelloShare', 'ac_Random-Channel'])
    transformed = transformer.fit(df)
    assert list(transformed.columns) == ['ac_refer_a_friend', 'ac_helloshare', 'ac_random_channel']


def test_column_name_transform():
    transformer = ColumnNameTransformer()
    df = pd.DataFrame(columns=['ac_Refer A Friend', 'ac_HelloShare', 'ac_Random-Channel'])
    transformed = transformer.transform(df)
    assert list(transformed.columns) == ['ac_refer_a_friend', 'ac_helloshare', 'ac_random_channel']
