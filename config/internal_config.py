import os

from dotenv import load_dotenv, find_dotenv

from voucher_opt.config.aws_parameters import AWSParameters
from voucher_opt.constants import Environment, S3_CORE_DATA_BUCKET, S3_MODELS_BUCKET, ATHENA_MODELS_DB, \
    ATHENA_MODELS_TABLE, \
    S3_CONFIG_BUCKET

ENVIRONMENT = Environment.get_env(os.getenv('ENVIRONMENT', default=Environment.DEVELOPMENT.name))

ENVIRONMENT_CONFIG = {
    Environment.DEVELOPMENT: {
        S3_CORE_DATA_BUCKET: 'bandit-data-hellofresh-staging',
        S3_MODELS_BUCKET: 'models-happy-hour-staging',
        S3_CONFIG_BUCKET: 'bandit-config-hellofresh-staging',
        ATHENA_MODELS_DB: 'happyhourmodelsstaging',
        ATHENA_MODELS_TABLE: 'model_output'
    },
    Environment.STAGING: {
        S3_CORE_DATA_BUCKET: 'bandit-data-hellofresh-staging',
        S3_MODELS_BUCKET: 'models-happy-hour-staging',
        S3_CONFIG_BUCKET: 'bandit-config-hellofresh-staging',
        ATHENA_MODELS_DB: 'happyhourmodelsstaging',
        ATHENA_MODELS_TABLE: 'model_output'
    },
    Environment.PRODUCTION: {
        S3_CORE_DATA_BUCKET: 'bandit-data-hellofresh',
        S3_MODELS_BUCKET: 'models-happy-hour',
        S3_CONFIG_BUCKET: 'bandit-config-hellofresh',
        ATHENA_MODELS_DB: 'happyhourmodels',
        ATHENA_MODELS_TABLE: 'model_output'
    }
}[ENVIRONMENT]

load_dotenv(find_dotenv())

# AWS secket access keys, must be environment variables
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = 'eu-west-1'
ATHENA_S3_STAGING_DIR = 's3://happy-hour-s3-staging-dir'

EVENT_TABLE_IDENTIFIER = 'fact_tables.ds_bandit_experiments'

PROJECT_CONFIG_PATH = 'config/project_config.toml'

if ENVIRONMENT != ENVIRONMENT.PRODUCTION:
    os.environ['INFLUXDB_DATABASE'] = 'test'

AWS_PARAMETERS: AWSParameters = AWSParameters(ENVIRONMENT_CONFIG)

# Numerical transformation parameters
NUMBER_OF_BINS = 10

# Categorical transformation parameters
CATEGORICAL_VALUE_RATIO_THRESH = 0.05
