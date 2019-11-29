import pandas as pd

from voucher_opt.features.feature_definitions import Feature
from voucher_opt.logger import log
from voucher_opt.transformers.transformer import Transformer


class CategoricalTransformer(Transformer):
    def __init__(self, categorical_features: [Feature], categorical_value_ratio_thresh):
        self._categorical_features: [Feature] = categorical_features
        self._categorical_value_ratio_thresh = categorical_value_ratio_thresh
        self._feature_columns = None

    def fit(self, df: pd.DataFrame):
        input_feature_columns = [feat.short_name for feat in self._categorical_features]
        feature_df = df[input_feature_columns].copy()
        for feature in self._categorical_features:
            log.debug(f'Transforming feature: {feature.original_name}')
            feature_df.loc[:, feature.short_name] = self._filter_categories(feature_df[feature.short_name])
        feature_df = pd.get_dummies(feature_df, columns=input_feature_columns).fillna(0)
        self._feature_columns = feature_df.columns
        df = pd.concat([df.drop(input_feature_columns, axis=1), feature_df], axis=1)
        return df

    def transform(self, df: pd.DataFrame):
        if df.empty:
            return df
        input_feature_columns = [feat.short_name for feat in self._categorical_features]
        feature_df = df[input_feature_columns]
        feature_df = pd.get_dummies(feature_df, columns=[feat.short_name for feat in self._categorical_features])
        feature_df = feature_df.reindex(columns=self._feature_columns, fill_value=0)
        df = pd.concat([df.drop(input_feature_columns, axis=1), feature_df], axis=1)
        return df

    def _filter_categories(self, filtered_col):
        value_counts = filtered_col.value_counts(normalize=True)
        filtered_col[~filtered_col.isin(value_counts[value_counts > self._categorical_value_ratio_thresh].index)] = None
        return filtered_col
