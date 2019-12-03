from abc import abstractmethod

import numpy as np
import pandas as pd

from voucher_opt.constants import ACTION_CODE, SEGMENT, LOGPROB, ACTION_PROB, ACTION_SAMPLES, \
    ACTION_REWARD, ACTION_IDX, GIVER_ID, UNIVERSE
from voucher_opt.logger import log
from voucher_opt.transformers.transformer import Transformer

ACTION_DISTRIBUTION = 'action_distribution'

FITTED_COLUMNS = [SEGMENT, ACTION_DISTRIBUTION]


class ContextualBandit(Transformer):
    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _sample_actions(self, df) -> pd.DataFrame:
        pass


class EpsGreedyBandit(ContextualBandit):
    def __init__(self, default_action, epsilon, actions, action_confidence_threshold, model_probability):
        self._default_action = default_action
        self._epsilon = epsilon
        self._available_actions = actions
        self._action_confidence_threshold = action_confidence_threshold
        self._model_probability = model_probability
        self._eps_greedy_distribution = pd.DataFrame()

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df = pd.DataFrame([[UNIVERSE, list(uniform_distribution(self._available_actions))]],
                              columns=FITTED_COLUMNS)
        else:
            df = df.groupby(SEGMENT).apply(self._calculate_eps_greedy_distribution, self._epsilon).reset_index(
                drop=True)
            df = df[[SEGMENT, ACTION_CODE, ACTION_PROB]].pivot(
                index=SEGMENT,
                columns=ACTION_CODE,
                values=ACTION_PROB
            )
            df[ACTION_DISTRIBUTION] = df.astype(float).values.tolist()

        df = df.reset_index()[FITTED_COLUMNS]
        self._eps_greedy_distribution = df
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._sample_actions(df)
        return df

    def _sample_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        df = pd.merge(df[[GIVER_ID, SEGMENT]], self._eps_greedy_distribution, how='left', on=SEGMENT)

        log.debug('Sampling actions from customer distributions...')
        df.loc[:, ACTION_IDX] = df.apply(_select_action, axis=1)
        log.debug('Getting action codes...')
        df.loc[:, ACTION_CODE] = df[ACTION_IDX].apply(lambda x: self._available_actions[x])
        log.debug('Computing logprobs...')
        df.loc[:, LOGPROB] = df.apply(self._logprob, axis=1)

        return df

    def _calculate_eps_greedy_distribution(self, segment, epsilon=0.1):
        actions = segment[ACTION_CODE].values
        segment = self._add_missing_actions(actions, segment)
        segment = segment.sort_values(by=ACTION_CODE)
        aggregated_rewards = segment[ACTION_REWARD].values
        num_actions = len(self._available_actions)
        if np.all(segment[ACTION_SAMPLES] >= self._action_confidence_threshold):
            probs = np.ones([num_actions]) * epsilon / (num_actions - 1)
            probs[np.argmax(aggregated_rewards)] = 1 - epsilon
        else:
            probs = _default_action_distr(self._default_action, self._available_actions)
        segment[ACTION_PROB] = probs
        return segment

    def _add_missing_actions(self, actions, segment):
        missing_actions = (set(self._available_actions) - set(actions))
        if len(missing_actions) > 0:
            rows_to_add = [(segment[SEGMENT].values[0], action, 0.0, 0) for action in missing_actions]
            segment = pd.concat([segment, pd.DataFrame(rows_to_add, columns=segment.columns)])
        return segment

    def _logprob(self, row):
        logprob = self._model_probability * row[ACTION_DISTRIBUTION][row[ACTION_IDX]]
        return logprob


def uniform_distribution(actions):
    num_actions = len(actions)
    return np.ones([num_actions]) / num_actions


def _default_action_distr(default_action, available_actions):
    default_action_idx = np.where(np.array(available_actions) == default_action)[0][0]
    probs = np.zeros(len(available_actions))
    probs[default_action_idx] = 1.0
    return probs


def _select_action(row):
    action_idx = np.random.choice(len(row[ACTION_DISTRIBUTION]), p=(row[ACTION_DISTRIBUTION]))
    return action_idx
