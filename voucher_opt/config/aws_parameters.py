from typing import Dict

from config import internal_config


class AWSParameters:
    def __init__(self, environment_config: Dict):
        self.access_key = internal_config.AWS_ACCESS_KEY_ID
        self.secret = internal_config.AWS_SECRET_ACCESS_KEY
        self.region = internal_config.AWS_REGION
        self.athena_s3_staging_dir = internal_config.ATHENA_S3_STAGING_DIR
        self.s3_core_data_bucket = environment_config[internal_config.S3_CORE_DATA_BUCKET]
        self.s3_models_bucket = environment_config[internal_config.S3_MODELS_BUCKET]
        self.s3_config_bucket = environment_config[internal_config.S3_CONFIG_BUCKET]
        self.athena_models_db = environment_config[internal_config.ATHENA_MODELS_DB]
        self.athena_models_table = environment_config[internal_config.ATHENA_MODELS_TABLE]
