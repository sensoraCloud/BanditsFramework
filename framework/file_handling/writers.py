import abc
import os
from pathlib import Path

import boto3

from config.internal_config import ENVIRONMENT, AWS_PARAMETERS
from voucher_opt.config.project_parameters import project_parameters
from voucher_opt.constants import Environment
from voucher_opt.logger import log


class Writer(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def write_dataframe(self, df, file_id, bucket, partition_params):
        pass

    @abc.abstractmethod
    def write_string(self, string, file_id, bucket, partition_params):
        pass


def _put_object_to_s3(body, location, filename):
    s3 = boto3.client('s3', aws_access_key_id=AWS_PARAMETERS.access_key, aws_secret_access_key=AWS_PARAMETERS.secret)
    s3.put_object(Bucket=location, Key=filename, Body=body)


def write_final_output(model_output_df, path):
    res = model_output_df.to_csv(index=False)
    _put_object_to_s3(res, AWS_PARAMETERS.s3_models_bucket, path)


class S3Writer(Writer):
    def write_dataframe(self, df, file_id, bucket, partition_params):
        res = df.to_csv(index=False)
        path = project_parameters.compile_path(partition_params, file_id, 'csv')
        log.info(f'Writing {file_id} to {path}')
        _put_object_to_s3(res, bucket, path)

    def write_string(self, string, file_id, bucket, partition_params):
        path = project_parameters.compile_path(partition_params, file_id, 'txt')
        log.info(f'Writing {file_id} to {path}')
        _put_object_to_s3(string, bucket, path)


class LocalWriter(Writer):
    def __init__(self):
        self._out_folder = 'dev_environment'

    def write_dataframe(self, df, file_id, bucket, partition_params):
        path = project_parameters.compile_path(partition_params, file_id, 'csv')
        local_path = self._create_local_path(bucket, path)
        log.info(f'Writing {file_id} to {local_path}')
        df.to_csv(local_path, index=False)

    def write_string(self, string, file_id, bucket, partition_params):
        path = project_parameters.compile_path(partition_params, file_id, 'txt')
        local_path = self._create_local_path(bucket, path)
        log.info(f'Writing {file_id} to {local_path}')
        with open(local_path, 'w') as f:
            f.write(string)

    def _create_local_path(self, bucket, path):
        local_path = Path(self._out_folder, bucket, path)
        os.makedirs(local_path.parent, exist_ok=True)
        return local_path


if ENVIRONMENT in (Environment.STAGING, Environment.PRODUCTION):
    writer = S3Writer()
else:
    writer = LocalWriter()
