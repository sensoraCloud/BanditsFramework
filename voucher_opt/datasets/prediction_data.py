import hfds

from config.internal_config import AWS_PARAMETERS
from voucher_opt.constants import COUNTRY_PARTITION_KEY, RUN_ID_PARTITION_KEY, \
    MODEL_ID_PARTITION_KEY, RAW_PREDICTION_DATA_CACHE_KEY, GIVER_ID
from voucher_opt.datasets.queries import compile_prediction_data_query
from voucher_opt.file_handling.cache import cacheable, partitions
from voucher_opt.file_handling.writers import writer
from voucher_opt.logger import log


def prediction_data_columns(features):
    return [GIVER_ID] + [feature.original_name for feature in features]


@cacheable(cache_key=RAW_PREDICTION_DATA_CACHE_KEY, partition_keys=[COUNTRY_PARTITION_KEY, RUN_ID_PARTITION_KEY])
def get_raw_prediction_data(country, ref_date, features):
    complete_prediction_data_query = compile_prediction_data_query(country, ref_date, features)

    writer.write_string(complete_prediction_data_query, 'prediction_data_query', AWS_PARAMETERS.s3_core_data_bucket,
                        partitions(COUNTRY_PARTITION_KEY, MODEL_ID_PARTITION_KEY))

    log.info('Fetching data for customers which will be assigned actions...')
    unprepared_prediction_df = hfds.db.run_dwh_query(complete_prediction_data_query)
    unprepared_prediction_df = unprepared_prediction_df.drop_duplicates(subset=[GIVER_ID])

    return unprepared_prediction_df[prediction_data_columns(features)]
