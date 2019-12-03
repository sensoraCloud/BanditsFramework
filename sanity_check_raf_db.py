import os

import click
import psycopg2
import s3fs
from dotenv import load_dotenv, find_dotenv

from config.internal_config import ENVIRONMENT
from voucher_opt.constants import COUNTRY, CONTROL, ACTION_CODE, ACTION_GENERATED_DATE, EXPLORATION, LOGPROB, \
    MODEL_VERSION, PROJECT, SEGMENT, ACTION_DESC
from tools.monitor_rok import validate_model_in_rok
from tools.monitor_utils import get_latest_model_ids, export_to_influx, plural_suffix

rok_DB_CONN_TRIES = 3

ACTION_DESCRIPTION = 'action_description'

rok_TO_PREDICTION_COLUMNS = {
    ACTION_CODE: ACTION_CODE,
    ACTION_DESCRIPTION: ACTION_DESC,
    ACTION_GENERATED_DATE: ACTION_GENERATED_DATE,
    CONTROL: CONTROL,
    COUNTRY: COUNTRY,
    EXPLORATION: EXPLORATION,
    LOGPROB: LOGPROB,
    MODEL_VERSION: MODEL_VERSION,
    PROJECT: PROJECT,
    SEGMENT: SEGMENT
}

load_dotenv(find_dotenv())


@click.command()
@click.argument('execution_date', type=str)
def main(execution_date):
    print("rok DB sanity check")
    print(f'environment = {ENVIRONMENT.name}')
    print(f'execution_date = {execution_date}')
    rok_conn = psycopg2.connect(dbname=os.getenv('rok_DB'), host=os.getenv('rok_HOST'), user=os.getenv('rok_USER'),
                                password=os.getenv('rok_PASSWORD'))
    s3 = s3fs.S3FileSystem()
    errors = {}

    latest_model_ids = get_latest_model_ids(execution_date, s3)
    for model_id in latest_model_ids:
        validate_model_in_rok(model_id, errors, rok_conn, s3)

    print()
    print(f'Found in total {sum(len(model_errors) for model_errors in errors.values())} '
          f'error{plural_suffix(errors)} related to the rok customer tracking table:\n')
    print()

    for model_id, model_errors in errors.items():
        if len(model_errors):
            print(f'Errors for {model_id}:')
            print('*' * 150)
            print(*model_errors, sep='\n')
            print('*' * 150 + '\n')

    export_to_influx('rok', errors, execution_date)


if __name__ == '__main__':
    main()
