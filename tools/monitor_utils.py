import functools
import re
from datetime import datetime

import hfds
import numpy as np
import pandas as pd
from hfds.config import INFLUXDB_HOST, INFLUXDB_DATABASE

from voucher_opt.constants import GIVER_ID


def get_latest_model_ids(execution_date, s3):
    keys = s3.glob(f's3://models-happy-hour/model_id=*')
    dates = set(key[33:43] for key in keys if key[33:43] <= execution_date)
    last_date = max(dates)
    datetime.strptime(last_date, '%Y-%m-%d')
    latest_model_ids = [re.search('model_id=(.*)', key).group()[9:] for key in keys if last_date in key]
    return latest_model_ids


def fetch_model_output_and_key(model_id, s3):
    key = s3.glob(f's3://models-happy-hour/model_id={model_id}/country=*/{model_id}.csv')[0]
    return pd.read_csv(f's3://{key}'), key


def check_consistency(merged_df, left_df_name, right_df_name, model_id, col_name, errors):
    model_errors = errors.setdefault(model_id, [])
    checked_df = merged_df.copy()
    left_col = f'{col_name}_x'
    right_col = f'{col_name}_y'
    if checked_df[left_col].dtype == np.float64 and checked_df[right_col].dtype == np.float64:
        checked_df = round_cols(left_col, right_col, checked_df)
    if checked_df[left_col].dtype != np.number and checked_df[right_col].dtype != np.number:
        checked_df = normalize_str_col(left_col, right_col, checked_df)
    inconsistent_values = checked_df[checked_df[left_col] != checked_df[right_col]][
        [GIVER_ID, left_col, right_col]].rename(index=str,
                                                columns={left_col: left_df_name, right_col: right_df_name})
    inconsistent_value_count = len(inconsistent_values)
    if inconsistent_value_count > 0:
        model_errors.append(
            f'There are {inconsistent_value_count} inconsistent values in "{col_name}" for {model_id}.'
            f'Examples:\n{inconsistent_values.head().to_string()}')


def round_cols(model_out_col, external_db_col, df):
    df.loc[:, model_out_col] = df[model_out_col].apply(functools.partial(round, ndigits=3))
    df.loc[:, external_db_col] = df[external_db_col].apply(functools.partial(round, ndigits=3))
    return df


def normalize_str_col(model_out_col, raf_col, df):
    df.loc[:, model_out_col] = df[model_out_col].astype(str).str.strip()
    df.loc[:, raf_col] = df[raf_col].astype(str).str.strip()
    return df


def export_to_influx(external_db_name, errors, execution_date):
    print(f'Exporting errors to InfluxDB, {INFLUXDB_HOST}:{INFLUXDB_DATABASE}')
    grafana_points = []
    for model_id, model_errors in errors.items():
        grafana_points.append({
            'measurement': f'happy_{external_db_name}_errors',
            'tags': {
                'country': model_id[17:19]
            },
            'fields': {
                'value': len(model_errors)
            },
            'time': datetime.strptime(execution_date, '%Y-%m-%d').isoformat()
        })
    hfds.grafana.send_to_influxdb(grafana_points)
    print(f'{len(grafana_points)} points exported.')


def plural_suffix(values):
    return 's' if len(values) != 1 else ''
