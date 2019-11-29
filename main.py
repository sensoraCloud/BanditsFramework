import os
from datetime import datetime

import click
from raven import Client

from config import internal_config
from config.internal_config import PROJECT_CONFIG_PATH
from voucher_opt.config.model_parameters import ModelParameters
from voucher_opt.config.global_parameters import GlobalParameters
from voucher_opt.config.project_parameters import load_project_parameters
from voucher_opt.constants import Environment
from voucher_opt.file_handling.cache import parse_custom_args
from voucher_opt.file_handling.cloud_utils import load_run_config_from_s3
from voucher_opt.file_handling.file_utils import load_run_config_from_local
from voucher_opt.voucher_model import run_model

if internal_config.ENVIRONMENT == Environment.PRODUCTION:
    client = Client(os.getenv('SENTRY_URL'))


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True
))
@click.argument('country')
@click.option('--prediction_date', default=datetime.now().strftime('%Y-%m-%d'), help='Date to run predictions for.')
@click.option('--run_config', default=None, help='Path of local run config. If not set, config is loaded from S3.',
              type=str)
@click.pass_context
def main(ctx, country, prediction_date, run_config):
    parse_custom_args(ctx)

    country = country.upper()
    load_project_parameters(PROJECT_CONFIG_PATH, country)
    model_parameters, global_parameters = load_parameters_from_run_config(country, run_config)

    run_model(country, prediction_date, internal_config.ENVIRONMENT, model_parameters, global_parameters)


def load_parameters_from_run_config(country, run_config):
    if run_config:
        config = load_run_config_from_local(country, run_config)
    else:
        config = load_run_config_from_s3(country)
    model_parameters = ModelParameters.create_from_run_config(config)
    global_parameters = GlobalParameters.create_from_run_config(config)
    return model_parameters, global_parameters


if __name__ == '__main__':
    main()
