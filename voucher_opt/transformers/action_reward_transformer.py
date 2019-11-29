import pandas as pd

from voucher_opt.constants import ACTION_CODE, SEGMENT, ACTION_REWARD, ACTION_SAMPLES
from voucher_opt.errors import InvalidActionError
from voucher_opt.transformers.transformer import Transformer

COLUMNS = [SEGMENT, ACTION_CODE, ACTION_REWARD, ACTION_SAMPLES]


class ActionRewardTransformer(Transformer):
    def __init__(self, default_action, actions, reward_agg_function):
        self._default_action = default_action
        self._available_actions = actions
        self._reward_agg_function = reward_agg_function

    def fit(self, df: pd.DataFrame):
        if df.empty:
            df = pd.DataFrame(columns=COLUMNS)
        else:
            df.groupby([SEGMENT, ACTION_CODE]) \
                .apply(self._validate_actions)

            reward = df.groupby([SEGMENT, ACTION_CODE]) \
                .apply(lambda x: self._reward_agg_function(x.feedback)) \
                .reset_index(name=ACTION_REWARD)

            samples = df.groupby([SEGMENT, ACTION_CODE]) \
                .apply(lambda x: len(x.feedback)) \
                .reset_index(name=ACTION_SAMPLES)

            df = pd.merge(reward, samples, how='inner', on=[SEGMENT, ACTION_CODE])

        df.columns = COLUMNS
        return df

    def transform(self, df: pd.DataFrame):
        return df

    def _validate_actions(self, group):
        invalid_actions = set(group[ACTION_CODE]) - set(self._available_actions)
        if len(invalid_actions) > 0:
            raise InvalidActionError(f'Found invalid action(s): {invalid_actions}.')
