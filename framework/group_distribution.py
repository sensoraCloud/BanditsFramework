import numpy as np

from voucher_opt.config.global_parameters import GlobalParameters
from voucher_opt.constants import CONTROL, EXPLORATION, LOGPROB, ACTION_CODE, SEGMENT
from voucher_opt.pipelines.bandit_agent import PREDICTION_COLUMNS


def set_experiment_groups(df, global_parameters: GlobalParameters):
    control_perc = global_parameters.control
    other_action_perc = global_parameters.other_action

    df[CONTROL] = 0
    df[EXPLORATION] = 0
    df[LOGPROB] = 0
    df[SEGMENT] = None

    no_control_perc = 1 - control_perc - other_action_perc
    df = _set_control_groups(df, control_perc, other_action_perc, global_parameters.default_action)
    df = _set_exploration_group(df, no_control_perc, global_parameters.exploration, global_parameters.actions)

    experimental_group_idxs = (df[CONTROL] == 0) & (df[EXPLORATION] == 0)
    experimental_group_df = df[experimental_group_idxs].drop([CONTROL, EXPLORATION, SEGMENT, LOGPROB, ACTION_CODE],
                                                             axis=1)
    other_groups_df = df[~experimental_group_idxs][PREDICTION_COLUMNS]

    return experimental_group_df, other_groups_df


def _set_control_groups(df, control_perc, other_action_perc, default_action):
    df = df.reset_index(drop=True)

    no_control_perc = 1 - control_perc - other_action_perc
    df[CONTROL] = np.random.choice([0, 1, 2],
                                   size=df.control.count(),
                                   p=[no_control_perc, control_perc, other_action_perc])
    df.loc[df[CONTROL] == 1, LOGPROB] = control_perc
    df.loc[df[CONTROL] == 2, LOGPROB] = other_action_perc
    df.loc[df[CONTROL] != 0, ACTION_CODE] = default_action

    return df


def _set_exploration_group(df, no_control_perc, exploration_perc, available_actions):
    df = df.reset_index(drop=True)

    not_control_customers_idxs = np.where(df[CONTROL] == 0)[0]
    num_customers_to_explore = int(len(not_control_customers_idxs) * exploration_perc)

    if num_customers_to_explore > 0:
        df = _assign_explore(df, exploration_perc, no_control_perc, not_control_customers_idxs,
                             num_customers_to_explore, available_actions)
    return df


def _assign_explore(df, exploration_perc, no_control_perc, not_control_customers_idxs, num_customers_to_explore,
                    available_actions):
    explore_idxs = np.random.choice(not_control_customers_idxs, size=num_customers_to_explore, replace=False)
    uniform_action_prob = (1 / len(available_actions))

    df.loc[explore_idxs, EXPLORATION] = 1
    df.loc[explore_idxs, LOGPROB] = no_control_perc * exploration_perc * uniform_action_prob
    df.loc[explore_idxs, ACTION_CODE] = np.random.choice(available_actions, num_customers_to_explore)

    return df
