import numpy as np
import pandas as pd

from voucher_opt.logger import log
from voucher_opt.transformers.transformer import Transformer


class NumericalFeatureTransformer(Transformer):
    def __init__(self, numerical_features, number_of_bins):
        self._numerical_features = numerical_features
        self._number_of_bins = number_of_bins
        self._feature_to_bins = {}

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self._numerical_features:
            feature_col = feature.short_name

            binned_feature = 'b_' + feature_col
            binned_feature_series, bins = pd.qcut(df[feature_col], q=self._number_of_bins, labels=False, retbins=True,
                                                  duplicates='drop', precision=1)
            if np.std(bins) < .01:
                df = df.drop([feature_col], axis=1)
                log.debug(
                    f'Feature {feature.original_name} was removed, the percentiles are approximately equal.')
                continue

            log.debug(f'Transforming feature: {feature.original_name}')

            df[binned_feature] = binned_feature_series

            self._feature_to_bins[feature_col] = bins
            log.debug(f'\nBin counts:\n'
                      f'{df[binned_feature].value_counts().sort_index()}\n'
                      f'{self._feature_to_bins[feature_col]}')
            df = df.drop([feature_col], axis=1)
            df = _one_hot_encode_bins(df, binned_feature, self._feature_to_bins[feature_col])
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        for feature in self._numerical_features:
            feature_col = feature.short_name
            try:
                bins = self._feature_to_bins[feature_col]
            except KeyError:
                df = df.drop([feature_col], axis=1)
                log.debug(f'Feature {feature.original_name} was removed during training.')
                continue

            log.debug(f'Transforming feature: {feature.original_name}')

            binned_feature = 'b_' + feature_col
            df[binned_feature] = pd.cut(df[feature_col], bins=bins, duplicates='drop', labels=False,
                                        precision=1)
            df[binned_feature] = df[binned_feature].fillna(-1).astype(int)
            df = df.drop([feature_col], axis=1)
            df = _one_hot_encode_bins(df, binned_feature, self._feature_to_bins[feature_col])
        return df


def _one_hot_encode_bins(df, binned_feature, bins):
    one_hot_encoded_df = pd.get_dummies(df, columns=[binned_feature])
    dummy_cols = _add_missing_columns(binned_feature, bins, one_hot_encoded_df)
    one_hot_encoded_df = _remove_nan_column(binned_feature, one_hot_encoded_df)
    one_hot_encoded_df = one_hot_encoded_df[sorted(list(dummy_cols))]
    return pd.concat([df, one_hot_encoded_df], axis=1).drop(binned_feature, axis=1)


def _add_missing_columns(binned_feature, bins, one_hot_encoded_df):
    dummy_cols = set([f'{binned_feature}_{i}' for i in range(len(bins) - 1)])
    columns_to_add = dummy_cols - set(one_hot_encoded_df.columns)
    for col in columns_to_add:
        one_hot_encoded_df.loc[:, col] = 0
        one_hot_encoded_df.loc[:, col] = one_hot_encoded_df.loc[:, col].astype(int)
    return dummy_cols


def _remove_nan_column(binned_feature, one_hot_encoded_df):
    nan_column = f'{binned_feature}_-1'
    if nan_column in one_hot_encoded_df.columns:
        one_hot_encoded_df = one_hot_encoded_df.drop(nan_column, axis=1)
    return one_hot_encoded_df
