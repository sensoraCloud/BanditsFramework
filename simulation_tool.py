import logging
import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime

import click
import numpy as np
import pandas as pd
from hfds.datetime import hf_week_add, hf_week_sub
from sklearn.model_selection import ParameterGrid

from voucher_opt.config import project_parameters
from voucher_opt.config.model_parameters import ModelParameters
from voucher_opt.constants import GIVER_ID, RECEIVER_ID, ACTION_CODE, FEEDBACK, LOGPROB, MATCHING_HF_WEEK, \
    SEGMENT, NON_FEATURE_COLUMNS
from voucher_opt.datasets.training_data import fetch_simulation_data, prepare_training_data, training_data_columns
from voucher_opt.features.feature_definitions import FeatureSet, feature_names
from voucher_opt.file_handling.cache import set_partition_parameters, parse_custom_args
from voucher_opt.pipelines.bandit_agent import MonkeyAgent, FixedAgent, EpsilonGreedyAgent, BanditAgent, \
    SegmentedEpsGreedyAgent

debug_log = logging.getLogger('main')
debug_log.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('simulation_debug_log.txt')
file_handler.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
debug_log.addHandler(file_handler)
debug_log.addHandler(stream_handler)


def create_debug_logger(logfile_path):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger = logging.getLogger('sim_tool')
    logger.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


sim_logger = create_debug_logger('simulation_log.txt')

EVALUATION_DIR = 'evaluation'

ACTUAL = 'actual'
PREDICTION = 'prediction'
PROB_ADJUSTED_FEEDBACK = 'prob_adjusted_feedback'
AGENT_NAME = 'agent_name'

# User configured parameters
EARLIEST_HF_WEEK = '2018-W40'
SIMULATION_NAME = 'seg_basic_params'

# Change this according to which actions exist in the dataset
ACTIONS = [0.0, 10.0, 20.0]

DEFAULT_ACTION = 0.0
REWARD_AGG_FUNC = "mean"
TREE_MIN_SAMPLE_CONF = 30
SEGMENT_NUM_TREES = 10
TREE_MAX_DEPTH = 6
TREE_CONF_BOUND = 0.01
EPSILON = 0.1
ACTION_CONFIDENCE_THRESHOLD = 50
NUMBER_OF_BINS = 4
CATEGORICAL_FREQUENCY_THRESHOLD = 0.05

model_parameters: ModelParameters = ModelParameters(EPSILON, ACTION_CONFIDENCE_THRESHOLD, TREE_MIN_SAMPLE_CONF,
                                                    SEGMENT_NUM_TREES, TREE_MAX_DEPTH, TREE_CONF_BOUND, NUMBER_OF_BINS,
                                                    CATEGORICAL_FREQUENCY_THRESHOLD, REWARD_AGG_FUNC)

EPS_GREEDY_PARAMS = {'epsilon': [0.01, 0.1, 0.2, 0.5]}

PARAMETER_GRID = ParameterGrid([{
    'epsilon': [0.01, 0.1, 0.2],
    'number_of_bins': [3, 5],
    'categorical_frequency_threshold': [0.1],
    'tree_min_sample_conf': [50, 200],
    'segment_num_trees': [10],
    'tree_max_depth': [2, 100],
    'tree_conf_bound': [0.00001]
}])


def create_feature_sets(country):
    feature_set: FeatureSet = FeatureSet.create_for_country(country)
    all_features = feature_set.all_features()
    feature_sets = [feature_set]
    if feature_set.has_experian_features():
        feature_set_without_experian = FeatureSet(
            [feature for feature in all_features
             if feature not in feature_set.experian_features()])
        feature_sets.append(feature_set_without_experian)
    return all_features, feature_sets


# Define which agents to evaluate here!
# Note: This method must be called for each iteration of the simulation, to create each agent from scratch.
def agents(all_features, feature_sets: [FeatureSet]):
    agents_to_evaluate = [MonkeyAgent(all_features, ACTIONS)] + \
                         [EpsilonGreedyAgent(all_features, ACTIONS, epsilon)
                          for epsilon in EPS_GREEDY_PARAMS['epsilon']] + \
                         [FixedAgent(all_features, action) for action in ACTIONS]

    agents_to_evaluate.extend(segmented_eps_greedy_agents(feature_sets))
    return agents_to_evaluate


def segmented_eps_greedy_agents(feature_sets):
    agents_to_add = []
    for feature_set in feature_sets:
        for parameters in PARAMETER_GRID:
            agents_to_add.append(
                SegmentedEpsGreedyAgent(feature_set, NON_FEATURE_COLUMNS, ACTIONS, DEFAULT_ACTION, 1.0,
                                        model_parameters, modified_parameters=parameters, evaluation_mode=True))
    return agents_to_add


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True
))
@click.argument('country')
@click.argument('num_feedback_weeks', type=click.INT)  # [3, 6, 10]
@click.argument('num_seed_weeks', type=click.INT)  # [2, 4]
@click.pass_context
def main(ctx, country, num_feedback_weeks, num_seed_weeks):
    np.random.seed(42)
    prediction_date = datetime.now().replace(microsecond=0)
    run_id = f'simulation-{country}-{prediction_date.isoformat()}'
    sim_logger.info(f'Evaluating: {SIMULATION_NAME}')
    sim_logger.info(f'Run ID: {run_id}')
    sim_logger.info(f'num_feedback_weeks={num_feedback_weeks}')
    sim_logger.info(f'num_seed_weeks={num_seed_weeks}')

    table_dir = f'{EVALUATION_DIR}/{SIMULATION_NAME}/{country}/' \
        f'feedback_weeks={num_feedback_weeks}/seed_weeks={num_seed_weeks}'

    initialize(ctx, run_id, country)

    all_features, feature_sets = create_feature_sets(country)

    dataset = load_dataset(country, prediction_date, num_feedback_weeks, EARLIEST_HF_WEEK, all_features)
    sim_logger.info(
        f'Number of agents to evaulate = {len(agents(all_features, feature_sets))}')

    agent_to_train_batch_idxs = defaultdict(lambda: pd.Index([]))
    agent_scores = pd.DataFrame([])

    hf_week_series = []
    hf_week = dataset[MATCHING_HF_WEEK].min()
    seed_weeks = [hf_week_add(hf_week, i) for i in range(num_seed_weeks)]
    all_matches = []
    all_actuals = []
    while hf_week in dataset[MATCHING_HF_WEEK].unique():
        if hf_week not in seed_weeks:
            hf_week_series.append(hf_week)
        agents_to_evaluate: [BanditAgent] = agents(all_features, feature_sets)
        sim_logger.info(f'Week {hf_week}')

        train_agents(agents_to_evaluate, dataset, agent_to_train_batch_idxs, hf_week, num_feedback_weeks)

        this_weeks_batch = dataset[dataset[MATCHING_HF_WEEK] == hf_week]
        if not (this_weeks_batch[ACTION_CODE] == 0.0).all():
            actual_df = this_weeks_batch[[GIVER_ID, RECEIVER_ID, FEEDBACK, ACTION_CODE, LOGPROB, MATCHING_HF_WEEK]] \
                .rename(index=str, columns={ACTION_CODE: ACTUAL}).copy()
            for agent in agents_to_evaluate:
                sim_logger.info(f'{hf_week}: Scoring agent {agent.name} with configuration {agent.configuration}')
                prediction_df = this_weeks_batch[[GIVER_ID] + feature_names(agent.features)].copy()
                matches = predict_and_match(agent, hf_week, prediction_df, actual_df)
                all_matches.append(matches)
                all_actuals.append(actual_df[[MATCHING_HF_WEEK, ACTUAL, FEEDBACK, LOGPROB]])
                if hf_week in seed_weeks:
                    train_batch = this_weeks_batch
                else:
                    train_batch = this_weeks_batch.reset_index().merge(matches[[GIVER_ID, RECEIVER_ID]],
                                                                       how='inner').set_index('index')
                    agent_scores = score_agent(agent, hf_week, matches, len(this_weeks_batch), agent_scores)
                train_batch_idxs = agent_to_train_batch_idxs[agent]
                agent_to_train_batch_idxs[agent] = train_batch_idxs.union(train_batch.index)
        if not agent_scores.empty:
            weekly_scores = agent_scores[agent_scores[MATCHING_HF_WEEK] == hf_week]
            write_df(f'{table_dir}/weekly', hf_week, run_id, weekly_scores)
        hf_week = hf_week_add(hf_week, 1)

    parameter_cols = list({p for g in PARAMETER_GRID for p in g}.union(EPS_GREEDY_PARAMS.keys()))
    write_df(table_dir, 'matches', run_id, pd.concat(all_matches, sort=False)[
        ['actual', 'agent_name', 'feedback', 'logprob', 'matching_hf_week', 'prediction', 'segment'] + parameter_cols])
    write_df(table_dir, 'actuals', run_id, pd.concat(all_actuals, sort=False))
    aggregate_final_scores(agent_scores, parameter_cols, run_id, table_dir)


def aggregate_final_scores(agent_scores, parameter_cols, run_id, table_dir):
    agent_scores = compute_cumulative_scores(agent_scores)
    agent_scores = agent_scores[
        [MATCHING_HF_WEEK, AGENT_NAME, 'dm_sum', 'dm_batch', 'dm_cum', 'ips_sum', 'ips_batch', 'ips_cum', 'hits',
         'num_samples'] + parameter_cols]
    write_df(table_dir, 'scores', run_id, agent_scores)


def initialize(ctx, run_id, country):
    parse_custom_args(ctx)
    project_parameters.load_project_parameters('config/evaluation_project_config.toml', country)
    set_partition_parameters(country, run_id)


def load_dataset(country, prediction_date, feedback_weeks, earliest_hf_week, features):
    dataset = fetch_simulation_data(country, prediction_date, feedback_weeks,
                                    project_parameters.project_parameters.model_version, features)
    dataset, metadata = prepare_training_data(features, prediction_date, dataset)
    dataset[RECEIVER_ID] = metadata[RECEIVER_ID]
    dataset[LOGPROB] = metadata[LOGPROB]
    dataset[MATCHING_HF_WEEK] = metadata[MATCHING_HF_WEEK]
    dataset = dataset[dataset[MATCHING_HF_WEEK] >= earliest_hf_week]
    if dataset.empty:
        sim_logger.error('Could not load any data. Closing.')
        sys.exit(1)
    sim_logger.info(f'Dataset shape = {dataset.shape}')
    dataset = dataset.sort_values(by=MATCHING_HF_WEEK)
    return dataset


def train_agents(agents_to_evaluate: [BanditAgent], dataset, agent_to_train_batch_idxs, hf_week, feedback_weeks):
    last_train_week = hf_week_sub(hf_week, feedback_weeks)
    for agent in agents_to_evaluate:
        try:
            idxs = agent_to_train_batch_idxs[agent]
            if not idxs.empty:
                train_batch = dataset.loc[idxs, :]
            else:
                train_batch = pd.DataFrame()
            sim_logger.info(f'{hf_week}: Training agent {agent.name} with configuration {agent.configuration} '
                            f'Training data shape = {train_batch.shape}')
            if train_batch.empty:
                train_batch = pd.DataFrame(
                    columns=training_data_columns(agent.features) + [MATCHING_HF_WEEK])
            train_batch = train_batch[train_batch[MATCHING_HF_WEEK] <= last_train_week]
            train_batch = train_batch[training_data_columns(agent.features)]
            agent.train(train_batch)
        except Exception:
            error = traceback.format_exc()
            sim_logger.error(error)
            sim_logger.error(
                f'Agent {agent.name} crashed while training during week {hf_week} with the following configuration:')
            sim_logger.error(str(agent.configuration))
            sys.exit(1)


def predict_and_match(agent, hf_week, prediction_df, actual_df):
    predictions = agent.predict(prediction_df)[[GIVER_ID, SEGMENT, ACTION_CODE]].rename(
        index=str, columns={ACTION_CODE: PREDICTION})
    rewards = actual_df.merge(predictions, on=GIVER_ID).drop_duplicates([GIVER_ID, RECEIVER_ID])
    matches = rewards[rewards.prediction == rewards.actual].copy()
    if not matches.empty:
        matches.loc[:, MATCHING_HF_WEEK] = hf_week
        matches.loc[:, AGENT_NAME] = agent.name
        for key, value in agent.configuration.items():
            matches.loc[:, key] = value
    return matches


def score_agent(agent: BanditAgent, hf_week, matches, num_samples, agent_scores):
    matches.loc[:, PROB_ADJUSTED_FEEDBACK] = (matches[FEEDBACK] / matches[LOGPROB])
    hits = 1 + len(matches)
    dm_sum = matches[FEEDBACK].sum()
    ips_sum = matches[PROB_ADJUSTED_FEEDBACK].sum()
    weekly_dm = dm_sum / hits
    weekly_ips = ips_sum / num_samples
    this_week_entry = [hf_week, agent.name, weekly_dm, dm_sum, hits, weekly_ips, ips_sum,
                       num_samples] + list(agent.configuration.values())
    weekly_scores = pd.DataFrame([this_week_entry],
                                 columns=[MATCHING_HF_WEEK, AGENT_NAME, 'dm_batch', 'dm_sum', 'hits', 'ips_batch',
                                          'ips_sum', 'num_samples'] + list(agent.configuration.keys()))
    agent_scores = pd.concat([agent_scores, weekly_scores], sort=False)
    return agent_scores


def weekly_scores_df(metric, agent_to_scores, hf_week_series):
    weekly_scores = pd.DataFrame(columns=[MATCHING_HF_WEEK] + list(agent_to_scores.keys()))
    weekly_scores[MATCHING_HF_WEEK] = hf_week_series
    for agent_name, agent_df in agent_to_scores.items():
        weekly_scores.loc[:, agent_name] = agent_df[f'{metric}_weekly'].values
    return weekly_scores


def compute_cumulative_scores(agent_scores):
    agent_scores['dm_cumsum'] = agent_scores.groupby(AGENT_NAME)[f'dm_sum'].cumsum()
    agent_scores['ips_cumsum'] = agent_scores.groupby(AGENT_NAME)[f'ips_sum'].cumsum()
    agent_scores['hits_cumsum'] = agent_scores.groupby(AGENT_NAME)['hits'].cumsum()
    agent_scores['num_samples_cumsum'] = agent_scores.groupby(AGENT_NAME)['num_samples'].cumsum()
    agent_scores['dm_cum'] = agent_scores[f'dm_cumsum'] / agent_scores[f'hits_cumsum']
    agent_scores['ips_cum'] = agent_scores[f'ips_cumsum'] / agent_scores[f'num_samples_cumsum']
    return agent_scores


def write_df(table_dir, table_name, run_id, df):
    os.makedirs(table_dir, exist_ok=True)
    file_path = f'{table_dir}/{table_name}_{run_id}.csv'
    sim_logger.info(f'Writing {file_path}')
    df.to_csv(file_path, index=False)


if __name__ == '__main__':
    main()
