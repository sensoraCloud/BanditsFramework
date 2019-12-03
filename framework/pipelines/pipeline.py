import pandas as pd

from voucher_opt.transformers.transformer import Transformer


class Pipeline:
    def __init__(self, *transformers: [Transformer]):
        self._transformers: [Transformer] = transformers

    def fit(self, df: pd.DataFrame):
        for transformer in self._transformers:
            df = transformer.fit(df)
        return df

    def transform(self, df: pd.DataFrame):
        for transformer in self._transformers:
            df = transformer.transform(df)
        return df
