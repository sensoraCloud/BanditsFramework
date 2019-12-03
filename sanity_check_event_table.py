import click
import s3fs
from dotenv import load_dotenv, find_dotenv

from config.internal_config import ENVIRONMENT
from tools.monitor_dwh import validate_models_in_dwh, fetch_event_data, send_stats_to_grokana
from tools.monitor_utils import export_to_influx, plural_suffix

rok_DB_CONN_TRIES = 3

load_dotenv(find_dotenv())


@click.command()
@click.argument('execution_date', type=str)
def main(execution_date):
    print("DWH event table sanity check")
    print(f'environment = {ENVIRONMENT.name}')
    print(f'execution_date = {execution_date}')
    s3 = s3fs.S3FileSystem()
    errors = {}

    print(f'Fetching DWH events...')
    event_df = fetch_event_data(execution_date)

    send_stats_to_grokana(event_df)

    validate_models_in_dwh(event_df, errors, s3)

    print()
    total_number_of_errors = sum(len(model_errors) for model_errors in errors.values())
    print(f'Found in total {total_number_of_errors} error{plural_suffix(errors)} '
          f'related to the DWH event table{"." if total_number_of_errors == 0 else ":"}\n')
    print()

    for model_id, model_errors in errors.items():
        if len(model_errors):
            print(f'Errors for {model_id}:')
            print('*' * 150)
            print(*model_errors, sep='\n')
            print('*' * 150 + '\n')

    export_to_influx('dwh', errors, execution_date)


if __name__ == '__main__':
    main()
