import boto3
import pandas as pd
import toml
from botocore.errorfactory import ClientError
from s3fs import S3FileSystem

from config.internal_config import AWS_PARAMETERS
from voucher_opt.config.project_parameters import project_parameters
from voucher_opt.logger import log
from voucher_opt.utils import load_config


class S3KeyNotFound(Exception):
    pass


def get_dataframe_from_s3(key, secret, s3_bucket, s3_key):
    try:
        s3 = boto3.client('s3', aws_access_key_id=key, aws_secret_access_key=secret)
        f = s3.get_object(Bucket=s3_bucket, Key=s3_key)

        df = pd.read_csv(f['Body'])
        return df
    except ClientError as e:
        raise S3KeyNotFound(e.response)


def load_run_config_from_s3(country):
    s3_key = project_parameters.compile_path({}, 'run_config', 'toml')
    log.info(f'Loading run config from {s3_key}')
    with S3FileSystem().open(f'{AWS_PARAMETERS.s3_config_bucket}/{s3_key}') as f:
        return load_config(toml.loads(f.read().decode()), country)
