from datetime import datetime

import plds
import numpy as np
import pandas as pd

from config.internal_config import AWS_PARAMETERS
from voucher_opt.config.global_parameters import GlobalParameters
from voucher_opt.config.model_parameters import ModelParameters
from voucher_opt.config.project_parameters import project_parameters
from voucher_opt.constants import NON_FEATURE_COLUMNS, ASSIGNED_DATE, \
    GIVER_ID
from voucher_opt.constants import RUN_CONFIG_FILE_ID, COUNTRY_PARTITION_KEY, \
    MODEL_ID_PARTITION_KEY, Environment, MODEL_OUTPUT_CACHE_KEY, MODEL_ID, RECEIVER_ID, COUNTRY, \
    ACTION_CODE, FEEDBACK, MATCHING_DATE, LOGPROB
from voucher_opt.datasets.prediction_data import get_raw_prediction_data
from voucher_opt.datasets.training_data import fetch_training_data, prepare_training_data
from voucher_opt.features.feature_definitions import FeatureSet
from voucher_opt.file_handling import writers
from voucher_opt.file_handling.cache import cacheable
from voucher_opt.file_handling.cache import set_partition_parameters, partitions
from voucher_opt.file_handling.writers import write_final_output
from voucher_opt.group_distribution import set_experiment_groups
from voucher_opt.logger import log
from voucher_opt.model_output import ModelOutput
from voucher_opt.monitoring import InfluxExporter
from voucher_opt.pipelines.bandit_agent import SegmentedEpsGreedyAgent
from voucher_opt.utils import random_str, print_header, print_footer
from voucher_opt.utils import string_seed


def run_model(country, prediction_date_str, environment, model_parameters: ModelParameters,
              global_parameters: GlobalParameters):
    prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d')
    model_id = f'model-{prediction_date_str}-{country}-{random_str(4)}'
    set_partition_parameters(country, model_id)
    _log_config(model_id, country, prediction_date_str, environment, global_parameters, model_parameters)

    influx_exporter = InfluxExporter(country, prediction_date, model_id)

    np.random.seed(string_seed(prediction_date_str))

    feature_set = FeatureSet.create_for_country(country)

    all_features = feature_set.all_features()
    training_df, training_meta_data = _get_training_data(country, prediction_date, all_features, global_parameters,
                                                         environment, influx_exporter)
    log.info('Training data value counts:')
    log.info(training_df.action_code.value_counts())

    log.info(f'Training data shape = {training_df.shape}')
    log.info(f'Training data columns = {list(training_df.columns)}')

    agent = SegmentedEpsGreedyAgent(feature_set, NON_FEATURE_COLUMNS, global_parameters.actions,
                                    global_parameters.default_action, global_parameters.experimental_group,
                                    model_parameters, training_meta_data)
    agent.train(training_df)

    prediction_df = get_raw_prediction_data(country, prediction_date_str, all_features)
    _validate_prediction_data(prediction_df)
    log.info(f'Prediction data shape = {prediction_df.shape}')
    log.info(f'Prediction data columns = {list(prediction_df.columns)}')
    prediction_df, other_groups_df = set_experiment_groups(prediction_df, global_parameters)
    log.info(f'Customers remaining after control and exploration = {len(prediction_df)}')

    predictions_df = agent.predict(prediction_df)

    model_output_df = _get_model_output(predictions_df, other_groups_df, country, prediction_date_str, model_id,
                                        global_parameters, influx_exporter)

    if environment == Environment.PRODUCTION:
        write_final_output(model_output_df, f'model_id={model_id}/country={country}/{model_id}.csv')


def _get_training_data(country, prediction_date, all_features, global_parameters: GlobalParameters, environment,
                       influx_exporter):
    unprepared_training_df = pd.DataFrame(
        columns=[GIVER_ID, ACTION_CODE, FEEDBACK, COUNTRY, ASSIGNED_DATE, RECEIVER_ID, MODEL_ID, MATCHING_DATE,
                 LOGPROB])
    if global_parameters.experimental_group > 0.0:
        unprepared_training_df = fetch_training_data(country, prediction_date, global_parameters.feedback_weeks,
                                                     all_features)
        if environment != Environment.DEVELOPMENT:
            _log_event_stats(unprepared_training_df, influx_exporter)
    else:
        log.info('Size of the experimental group is 0%. No events will be collected.')
    if unprepared_training_df.empty:
        log.info('Training data is empty. The actions will be selected uniformly at random.')

    training_df, training_meta_data = prepare_training_data(all_features, prediction_date, unprepared_training_df)

    return training_df, training_meta_data


def _log_event_stats(unprepared_data, influx_exporter):
    events_per_date = unprepared_data.copy().assigned_date.apply(
        lambda x: datetime.strptime(str(x), '%Y%m%d')).value_counts()
    influx_exporter.send_event_stats(events_per_date)
    events_per_week = events_per_date.reset_index(name='count')
    events_per_week.loc[:, 'index'] = events_per_week['index'].apply(plds.datetime.datetime_to_pl_week)
    log.debug('Events per pl week:')
    log.debug(events_per_week.groupby('index').sum().sort_index().to_string())


def _validate_prediction_data(prediction_df):
    assert len(prediction_df) > 100, 'The prediction query data returned less than 100 customers!'  # Rocco Code!!


@cacheable(cache_key=MODEL_OUTPUT_CACHE_KEY, partition_keys=[COUNTRY_PARTITION_KEY, MODEL_ID_PARTITION_KEY])
def _get_model_output(predictions_df, other_groups_df, country, elaboration_date_str, model_id,
                      global_parameters: GlobalParameters, influx_exporter):
    model_output = ModelOutput.create_from_transformed_data(predictions_df, other_groups_df)
    model_output.prepare_output(country, model_id, elaboration_date_str, global_parameters.action_desc)
    model_output.log_stats()
    influx_exporter.send_action_distribution(model_output.model_action_distribution())
    model_output_df = model_output.validate(global_parameters)
    return model_output_df


def _log_config(model_id, country, prediction_date, environment, global_parameters, model_parameters):
    print_header('CONFIG')
    config_str = _config_str(model_id, country, prediction_date, global_parameters, model_parameters, environment)
    log.info(config_str)
    print_footer()
    writers.writer.write_string(config_str,
                                RUN_CONFIG_FILE_ID,
                                AWS_PARAMETERS.s3_core_data_bucket,
                                partitions(COUNTRY_PARTITION_KEY, MODEL_ID_PARTITION_KEY))


def _config_str(model_id, country, prediction_date, global_parameters, model_parameters, environment):
    config_str = f'''
Project parameters:
    {project_parameters}
    model_id = {model_id}
    environment = {environment.name}
    country = {country}
    prediction_date = {prediction_date}
    s3_core_data_bucket = {AWS_PARAMETERS.s3_core_data_bucket}
    s3_models_bucket = {AWS_PARAMETERS.s3_models_bucket}
    athena_models_db = {AWS_PARAMETERS.athena_models_db}
    athena_models_table = {AWS_PARAMETERS.athena_models_table}
    athena_s3_staging_dir = {AWS_PARAMETERS.athena_s3_staging_dir}
Global parameters:
    {global_parameters}
Model parameters:
    {model_parameters}
'''
    return config_str
