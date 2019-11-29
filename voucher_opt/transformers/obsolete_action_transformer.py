import pandas as pd

from voucher_opt.constants import ACTION_CODE
from voucher_opt.transformers.transformer import Transformer


class ObsoleteActionTransformer(Transformer):
    def __init__(self, actions):
        self._actions = actions

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df[ACTION_CODE].isin(self._actions)]
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df
