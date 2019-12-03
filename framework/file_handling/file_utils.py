import toml

from voucher_opt.logger import log
from voucher_opt.utils import load_config


def load_run_config_from_local(country, path):
    log.info(f'Loading run config from {path}')
    run_config = toml.load(path)
    return load_config(run_config, country)


def read_query(query_path, **query_params):
    with open(query_path) as query_file:
        query = query_file.read().format(**query_params)
    return query
