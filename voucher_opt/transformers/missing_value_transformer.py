import pandas as pd

from voucher_opt.constants import HAS_EXPERIAN
from voucher_opt.features.feature_definitions import FeatureType, FeatureSet
from voucher_opt.transformers.transformer import Transformer

UNKNOWN_CATEGORICAL_VALUE = 'Unknown'


class MissingValueTransformer(Transformer):
    def __init__(self, feature_set: FeatureSet):
        self._feature_set = feature_set

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self._feature_set.has_experian_features():
            has_experian = ~df[[feature.original_name for feature in self._feature_set.experian_features()]].isna().any(
                axis=1)
            df.loc[:, HAS_EXPERIAN] = has_experian
        categorical_columns = [f.original_name for f in self._feature_set.categorical_features()]
        other_columns = [f.original_name for f in self._feature_set.all_features() if
                         f.f_type != FeatureType.CATEGORICAL]
        df[categorical_columns] = df[categorical_columns].fillna(UNKNOWN_CATEGORICAL_VALUE)
        df[other_columns] = df[other_columns].fillna(0)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df)
