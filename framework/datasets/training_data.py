from datetime import datetime

import plds
import pandas as pd
from plds.datetime import datetime_to_pl_week

from config import internal_config
from config.internal_config import AWS_PARAMETERS
from voucher_opt.config.project_parameters import project_parameters
from voucher_opt.constants import RECEIVER_ID, GIVER_ID, MODEL_ID, ASSIGNED_DATE, ELABORATION_DATE, \
    RAW_TRAINING_DATA_CACHE_KEY, \
    RUN_ID_PARTITION_KEY, COUNTRY_PARTITION_KEY, MODEL_ID_PARTITION_KEY, COUNTRY, MATCHING_DATE, LOGPROB, \
    MATCHING_pl_WEEK, ACTION_CODE, FEEDBACK
from voucher_opt.datasets.queries import compile_dataset_query
from voucher_opt.file_handling.cache import cacheable, partitions
from voucher_opt.file_handling.writers import writer
from voucher_opt.logger import log

TRAINING_META_DATA_COLUMNS = [COUNTRY, ASSIGNED_DATE, RECEIVER_ID, MODEL_ID, ELABORATION_DATE, MATCHING_DATE,
                              MATCHING_pl_WEEK, LOGPROB]


def training_data_columns(features):
    return [GIVER_ID, ACTION_CODE, FEEDBACK] + [feature.original_name for feature in features]


@cacheable(cache_key=RAW_TRAINING_DATA_CACHE_KEY, partition_keys=[COUNTRY_PARTITION_KEY, RUN_ID_PARTITION_KEY])
def fetch_training_data(country, prediction_date, feedback_weeks, features):
    dataset_query = compile_dataset_query(project_parameters.model_version, country, prediction_date, feedback_weeks,
                                          internal_config.EVENT_TABLE_IDENTIFIER, features)
    writer.write_string(dataset_query, 'dataset_query', AWS_PARAMETERS.s3_core_data_bucket,
                        partitions(COUNTRY_PARTITION_KEY, MODEL_ID_PARTITION_KEY))

    log.info('Fetching training data...')
    unprepared_training_df = plds.db.run_dwh_query(dataset_query)

    unprepared_training_df = unprepared_training_df.drop_duplicates(subset=[MODEL_ID, GIVER_ID, RECEIVER_ID])

    if unprepared_training_df.empty:
        log.info('No valid events found.')

    return unprepared_training_df


@cacheable(cache_key='simulation_dataset', partition_keys=[COUNTRY_PARTITION_KEY, RUN_ID_PARTITION_KEY])
def fetch_simulation_data(country, prediction_date, feedback_weeks, model_versions, features):
    print('Fetching dataset for simulation...')
    datasets = []
    for model_version in model_versions:
        dataset_query = compile_dataset_query(model_version, country, prediction_date, feedback_weeks,
                                              internal_config.EVENT_TABLE_IDENTIFIER, features)
        writer.write_string(dataset_query, 'dataset_query', AWS_PARAMETERS.s3_core_data_bucket,
                            partitions(COUNTRY_PARTITION_KEY, MODEL_ID_PARTITION_KEY))

        unprepared_training_df = plds.db.run_dwh_query(dataset_query)
        unprepared_training_df = unprepared_training_df.drop_duplicates(subset=[MODEL_ID, GIVER_ID, RECEIVER_ID])
        datasets.append(unprepared_training_df)

        if unprepared_training_df.empty:
            log.info('No valid events found.')

    dataset = pd.concat(datasets)
    return dataset


def prepare_training_data(features, prediction_date, unprepared_training_df):
    columns = training_data_columns(features)
    if unprepared_training_df.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame()
    unprepared_training_df.loc[:, ELABORATION_DATE] = datetime.strftime(prediction_date, '%Y-%m-%d')
    pl_week_series = unprepared_training_df[MATCHING_DATE].apply(lambda x: datetime.strptime(str(x), '%Y%m%d')).apply(
        datetime_to_pl_week)
    unprepared_training_df.loc[:, MATCHING_pl_WEEK] = pl_week_series
    training_meta_data = unprepared_training_df[[GIVER_ID] + TRAINING_META_DATA_COLUMNS].copy()
    training_meta_data.loc[:, ASSIGNED_DATE] = training_meta_data.assigned_date.apply(
        lambda x: datetime.strftime(datetime.strptime(str(x), '%Y%m%d'), '%Y-%m-%d'))
    training_df = unprepared_training_df[columns]
    return training_df, training_meta_data
