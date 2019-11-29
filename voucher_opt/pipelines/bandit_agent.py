import random
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd

from config.internal_config import NUMBER_OF_BINS, CATEGORICAL_VALUE_RATIO_THRESH
from voucher_opt.config.model_parameters import ModelParameters
from voucher_opt.constants import ACTION_CODE, LOGPROB, GIVER_ID, DATASET_CACHE_KEY, COUNTRY_PARTITION_KEY, \
    RUN_ID_PARTITION_KEY, \
    ELABORATION_DATE, ASSIGNED_DATE, RECEIVER_ID, MODEL_ID, FEEDBACK, SEGMENT, CONTROL, EXPLORATION, \
    PREDICTION_COLUMNS, ACTION_IDX
from voucher_opt.datasets.prediction_data import prediction_data_columns
from voucher_opt.datasets.training_data import training_data_columns
from voucher_opt.features.feature_definitions import FeatureSet
from voucher_opt.file_handling.cache import cacheable
from voucher_opt.logger import log
from voucher_opt.pipelines.pipeline import Pipeline
from voucher_opt.transformers.action_reward_transformer import ActionRewardTransformer
from voucher_opt.transformers.categorical_transformer import CategoricalTransformer
from voucher_opt.transformers.contextual_bandit import EpsGreedyBandit, uniform_distribution, ACTION_DISTRIBUTION
from voucher_opt.transformers.missing_value_transformer import MissingValueTransformer
from voucher_opt.transformers.numerical_transformer import NumericalFeatureTransformer
from voucher_opt.transformers.obsolete_action_transformer import ObsoleteActionTransformer
from voucher_opt.transformers.segmentation.tree_segment_transformer import TreeSegmentTransformer
from voucher_opt.transformers.transformer import FeatureNameTransformer, ColumnNameTransformer

DATASET_COLUMNS = [ELABORATION_DATE, ASSIGNED_DATE, GIVER_ID, RECEIVER_ID, MODEL_ID, ACTION_CODE, FEEDBACK, SEGMENT]


class BanditAgent(ABC):
    def __init__(self, features, modified_parameters=None):
        self._features = features
        if modified_parameters:
            self._modified_parameters = modified_parameters
        else:
            self._modified_parameters = {}

    @property
    def configuration(self):
        return OrderedDict(sorted(self._modified_parameters.items()))

    @property
    def features(self):
        return self._features

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _train(self, train_df):
        pass

    @abstractmethod
    def _predict(self, predict_df):
        pass

    def train(self, train_df):
        assert sorted(list(train_df.columns)) == sorted(training_data_columns(self._features))
        return self._train(train_df)

    def predict(self, predict_df):
        predict_df = predict_df.copy()
        assert sorted(list(predict_df.columns)) == sorted(prediction_data_columns(self._features))
        if predict_df.empty:
            return pd.DataFrame(columns=PREDICTION_COLUMNS)
        predict_df = self._predict(predict_df)
        predict_df.loc[:, GIVER_ID] = predict_df[GIVER_ID].astype(int)
        predict_df.loc[:, CONTROL] = 0
        predict_df.loc[:, EXPLORATION] = 0
        return predict_df[PREDICTION_COLUMNS]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, BanditAgent):
            return self.name == other.name and self.configuration == other.configuration
        return False

    def __hash__(self):
        return hash(self.name) + hash(tuple(sorted(self.configuration.items())))


class SegmentedEpsGreedyAgent(BanditAgent):
    def __init__(self, feature_set: FeatureSet, non_feature_columns, actions, default_action, model_probability,
                 model_parameters: ModelParameters, training_meta_data=pd.DataFrame(), modified_parameters=None,
                 evaluation_mode=False):
        super(SegmentedEpsGreedyAgent, self).__init__(feature_set.all_features(), modified_parameters)
        self._create_name(feature_set.has_experian_features())
        self._modified_parameters = modified_parameters
        if modified_parameters:
            model_parameters.set_from_dict(modified_parameters)
        self._training_meta_data = training_meta_data
        self._evaluation_mode = evaluation_mode
        self._feature_pipeline = Pipeline(
            ObsoleteActionTransformer(actions),
            MissingValueTransformer(feature_set),
            FeatureNameTransformer(feature_set.all_features()),
            NumericalFeatureTransformer(feature_set.numerical_features(), NUMBER_OF_BINS),
            CategoricalTransformer(feature_set.categorical_features(), CATEGORICAL_VALUE_RATIO_THRESH),
            ColumnNameTransformer(),
            TreeSegmentTransformer(non_feature_columns, actions, feature_set.has_experian_features(), model_parameters)
        )
        self._model_pipeline = Pipeline(
            ActionRewardTransformer(default_action, actions, model_parameters.reward_agg_function),
            EpsGreedyBandit(default_action, model_parameters.epsilon, actions,
                            model_parameters.action_confidence_threshold, model_probability)
        )

    @property
    def name(self):
        return self._name

    def _create_name(self, has_experian):
        self._name = f'Segmented-eps-greedy'
        self._name += f'_has_experian={has_experian}'

    def _train(self, train_df):
        if self._evaluation_mode:
            if not train_df.empty:
                train_df = self._feature_pipeline.fit(train_df)
            else:
                train_df = pd.DataFrame(columns=[ACTION_CODE, FEEDBACK, SEGMENT])
        else:
            train_df = self._transform_training_data(train_df)[[ACTION_CODE, FEEDBACK, SEGMENT]]
        train_df = self._model_pipeline.fit(train_df)
        return train_df

    @cacheable(cache_key=DATASET_CACHE_KEY, partition_keys=[COUNTRY_PARTITION_KEY, RUN_ID_PARTITION_KEY])
    def _transform_training_data(self, train_df):
        if not train_df.empty:
            train_df = self._feature_pipeline.fit(train_df)
            train_df = train_df.merge(self._training_meta_data, on=GIVER_ID)[DATASET_COLUMNS]
        else:
            train_df = pd.DataFrame(columns=DATASET_COLUMNS)
        return train_df

    def _predict(self, predict_df):
        predict_df = self._feature_pipeline.transform(predict_df)
        predict_df = predict_df[[GIVER_ID, SEGMENT]]
        predict_df = self._model_pipeline.transform(predict_df)

        log.info(f'Segments in prediction data: {", ".join(predict_df.segment.unique())}')

        return predict_df


class MonkeyAgent(BanditAgent):
    def __init__(self, features, actions):
        super(MonkeyAgent, self).__init__(features)
        self._actions = actions

    @property
    def name(self):
        return 'Monkey'

    def _train(self, train_df):
        return train_df

    def _predict(self, predict_df):
        predict_df.loc[:, ACTION_CODE] = predict_df.apply(lambda _: random.choice(self._actions), axis=1)
        predict_df.loc[:, LOGPROB] = 1.0 / len(self._actions)
        predict_df.loc[:, SEGMENT] = None
        return predict_df[[GIVER_ID, ACTION_CODE, LOGPROB, SEGMENT]]


class FixedAgent(BanditAgent):
    def __init__(self, features, action):
        super(FixedAgent, self).__init__(features)
        self._action = action

    @property
    def name(self):
        return f'Fixed ({self._action})'

    def _train(self, train_df):
        return train_df

    def _predict(self, predict_df):
        predict_df.loc[:, ACTION_CODE] = self._action
        predict_df.loc[:, LOGPROB] = 1.0
        predict_df.loc[:, SEGMENT] = None
        return predict_df[[GIVER_ID, ACTION_CODE, LOGPROB, SEGMENT]]


class EpsilonGreedyAgent(BanditAgent):
    def __init__(self, features, actions, epsilon):
        super(EpsilonGreedyAgent, self).__init__(features, {'epsilon': epsilon})
        self._actions = actions
        self._epsilon = epsilon
        self._probs = uniform_distribution(self._actions)

    @property
    def name(self):
        return f'Epsilon-greedy'

    def _train(self, train_df):
        if not train_df.empty:
            aggregated_rewards = train_df.groupby(ACTION_CODE)[[FEEDBACK]].mean()
            num_actions = len(self._actions)
            probs = np.ones([num_actions]) * self._epsilon / (num_actions - 1)
            probs[np.argmax(aggregated_rewards.values)] = 1 - self._epsilon
            self._probs = probs
        return train_df

    def _predict(self, predict_df):
        predict_df = predict_df.copy().reset_index(drop=True)
        predict_df.loc[:, ACTION_DISTRIBUTION] = pd.Series([list(self._probs)] * len(predict_df))
        predict_df.loc[:, ACTION_IDX] = predict_df.apply(lambda _: np.random.choice(len(self._probs), p=self._probs),
                                                         axis=1)
        predict_df.loc[:, ACTION_CODE] = predict_df[ACTION_IDX].apply(lambda x: self._actions[x])
        predict_df.loc[:, LOGPROB] = predict_df.apply(lambda x: x[ACTION_DISTRIBUTION][x[ACTION_IDX]], axis=1)
        predict_df.loc[:, SEGMENT] = None
        return predict_df[[GIVER_ID, ACTION_CODE, LOGPROB, SEGMENT]]
