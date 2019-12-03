import hashlib
import random
import string

import numpy as np

from voucher_opt.logger import log


def print_header(heading):
    log.info(heading)
    log.info('=' * 50)


def print_footer():
    log.info('=' * 50)


def random_str(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def string_seed(text):
    return np.frombuffer(hashlib.sha1(text.encode()).digest(), dtype='uint32')


def get_feature_columns(df, non_feature_columns):
    return list(df.columns.drop(list(set(non_feature_columns) & set(df.columns))))


def load_config(config, country):
    try:
        return dict(config['DEFAULT_CONFIG'], **config['COUNTRY_CONFIG'][country])
    except KeyError:
        return config['DEFAULT_CONFIG']


def validate_bounds(value_name, value, lower_bound=None, upper_bound=None):
    if lower_bound is not None:
        assert lower_bound <= value, f'The value of "{value_name}" is lower than {lower_bound}'
    if upper_bound is not None:
        assert value <= upper_bound, f'The value of "{value_name}" is higher than {upper_bound}'


def validate_type(variable_name, variable, expected_type):
    actual_type = type(variable)
    assert actual_type == expected_type, \
        f'{variable_name} has the wrong type. Expected type: "{expected_type}". Actual type = "{actual_type}"'
