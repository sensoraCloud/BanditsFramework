from abc import ABC, abstractmethod

import pandas as pd

from voucher_opt.features.feature_definitions import shorten_feature_names


class Transformer(ABC):

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


class FeatureNameTransformer(Transformer):
    def __init__(self, features):
        self._features = features

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_columns = [feature.original_name for feature in self._features]
        df = shorten_feature_names(df, feature_columns)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df)


class ColumnNameTransformer(Transformer):
    def fit(self, df: pd.DataFrame):
        df.columns = [x.lower().replace(' ', '_').replace('-', '_') for x in df.columns]
        return df

    def transform(self, df: pd.DataFrame):
        return self.fit(df)
