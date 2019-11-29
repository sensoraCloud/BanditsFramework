import functools
from collections import OrderedDict

import pandas as pd

from config.internal_config import AWS_PARAMETERS
from voucher_opt.constants import COUNTRY_PARTITION_KEY, RUN_ID_PARTITION_KEY, MODEL_ID_PARTITION_KEY
from voucher_opt.errors import InvalidCustomArgumentException
from voucher_opt.file_handling.writers import writer
from voucher_opt.logger import log

cache_paths = {}
partition_parameters = {}


def set_partition_parameters(country, model_id):
    partition_parameters[COUNTRY_PARTITION_KEY] = country
    partition_parameters[RUN_ID_PARTITION_KEY] = model_id
    partition_parameters[MODEL_ID_PARTITION_KEY] = model_id


def partitions(*partition_keys):
    return OrderedDict(
        {partition_key: partition_parameters[partition_key] for partition_key in partition_keys})


def cacheable(cache_key, partition_keys):
    def decorator_local_cache(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                path = cache_paths[cache_key]
                log.info(f'Loading pre-compiled {cache_key} from: {path}')
                return pd.read_csv(path)
            except KeyError:
                df = func(*args, **kwargs)
                writer.write_dataframe(df, cache_key, AWS_PARAMETERS.s3_core_data_bucket, partitions(*partition_keys))
                return df

        return wrapper

    return decorator_local_cache


def parse_custom_args(ctx):
    try:
        if [ctx.args[i] for i in range(0, len(ctx.args), 2) if not ctx.args[i].startswith('--')]:
            raise InvalidCustomArgumentException('Custom key-value arguments must be prepended with "--".')
        global cache_paths
        cache_paths = {ctx.args[i][2:]: ctx.args[i + 1] for i in range(0, len(ctx.args), 2)}
    except IndexError:
        raise InvalidCustomArgumentException(f'Custom key-value argument without value: "{ctx.args[-1]}". '
                                             f'Key and value of custom arguments should be space-separated.')
