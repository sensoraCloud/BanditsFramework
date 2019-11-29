from enum import Enum


class Environment(Enum):
    DEVELOPMENT = 'development'
    STAGING = 'staging'
    PRODUCTION = 'production'

    @classmethod
    def get_env(cls, environment):
        return {
            'DEV': cls.DEVELOPMENT,
            'DEVELOPMENT': cls.DEVELOPMENT,
            'STAGING': cls.STAGING,
            'PRODUCTION': cls.PRODUCTION
        }[environment.upper()]


S3_CORE_DATA_BUCKET = 'S3_CORE_DATA_BUCKET'
S3_MODELS_BUCKET = 'S3_MODELS_BUCKET'
S3_CONFIG_BUCKET = 'S3_CONFIG_BUCKET'
ATHENA_MODELS_DB = 'ATHENA_MODELS_DB'
ATHENA_MODELS_TABLE = 'ATHENA_MODELS_TABLE'

# Global parameters
ACTIONS_KEY = 'actions'
DEFAULT_ACTION_KEY = 'default_action'
ACTION_DESC_KEY = 'action_desc'
CONTROL_KEY = 'control'
OTHER_ACTION_KEY = 'other_action'
EXPLORATION_KEY = 'exploration'
FEEDBACK_WEEKS_KEY = 'feedback_weeks'

# Model parameters
EPSILON_KEY = 'epsilon'
ACTION_CONFIDENCE_THRESHOLD_KEY = 'action_confidence_threshold'
REWARD_AGG_FUNC_KEY = 'reward_agg_func'
TREE_MIN_SAMPLE_CONF_KEY = 'tree_min_sample_conf'
SEGMENT_NUM_TREES_KEY = 'segment_num_trees'
TREE_MAX_DEPTH_KEY = 'tree_max_depth'
TREE_CONF_BOUND_KEY = 'tree_conf_bound'
NUMBER_OF_BINS_KEY = 'number_of_bins'
CATEGORICAL_FREQUENCY_THRESHOLD_KEY = 'categorical_frequency_threshold'
HAS_EXPERIAN_KEY = 'has_experian'

# Cache keys
# Note: AWS Glue crawlers depend on these values. Don't change them!
DATASET_CACHE_KEY = 'dataset'
PREDICTION_DATA_CACHE_KEY = 'prediction_data'
RAW_TRAINING_DATA_CACHE_KEY = 'raw_training_data'
RAW_PREDICTION_DATA_CACHE_KEY = 'raw_prediction_data'
MODEL_OUTPUT_CACHE_KEY = 'model_output'

# Column names
ACTION_CODE = 'action_code'
ACTION_CONFIDENT = 'action_confident'
ACTION_DESC = 'action_desc'
ACTION_GENERATED_DATE = 'action_generated_date'
ACTION_PROB = 'action_prob'
ACTION_REDEEMED_DATETIME = 'action_redeemed_datetime'
ASSIGNED_DATE = 'assigned_date'
AVG_FEEDBACK = 'avg_feedback'
CONTROL = 'control'
COUNTRY = 'country'
ELABORATION_DATE = 'elaboration_date'
EXPLORATION = 'exploration'
FEEDBACK = 'feedback'
LOGPROB = 'logprob'
GIVER_ID = 'giver_id'
MODEL_ID = 'model_id'
MODEL_VERSION = 'model_version'
PROJECT = 'project'
RECEIVER_ID = 'receiver_id'
RUN_ID = 'run_id'
SEGMENT = 'segment'
ACTION_REWARD = 'action_reward'
ACTION_SAMPLES = 'action_samples'
ACTION_IDX = 'action_idx'
MATCHING_DATE = 'matching_date'
MATCHING_HF_WEEK = 'matching_hf_week'
HAS_EXPERIAN = 'has_experian'

# Partition keys
COUNTRY_PARTITION_KEY = 'country'
MODEL_ID_PARTITION_KEY = 'model_id'
RUN_ID_PARTITION_KEY = 'run_id'

# File IDs
RUN_CONFIG_FILE_ID = 'run_config'
MODEL_DEFINITION_FILE_ID = 'model_definition'
MODEL_OUTPUT_FILE_ID = 'model_output'

# Column lists
NON_FEATURE_COLUMNS = [COUNTRY, RUN_ID, ELABORATION_DATE, ASSIGNED_DATE, GIVER_ID, RECEIVER_ID, MODEL_ID, ACTION_CODE,
                       LOGPROB, FEEDBACK, SEGMENT, MATCHING_HF_WEEK, MATCHING_DATE]

PREDICTION_COLUMNS = [GIVER_ID, SEGMENT, ACTION_CODE, LOGPROB, CONTROL, EXPLORATION]

# Special values
UNIVERSE = 'UNIVERSE'
