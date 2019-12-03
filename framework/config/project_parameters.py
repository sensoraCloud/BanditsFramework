from pathlib import Path
from typing import Dict

import toml

from voucher_opt.utils import load_config


class ProjectParameters:
    def __init__(self):
        # Default parameters
        self.project_name = 'happy_hour'
        self.model_version = 2
        self.event_query = 'queries/event_query.txt'
        self.feedback_query = 'queries/feedback_query.txt'
        self.features_query = 'queries/features_query.txt'
        self.main_training_data_query = 'queries/main_training_data_query.txt'
        self.prediction_customers_query_path = 'queries/prediction_customers_query.txt'
        self.main_prediction_data_query_path = 'queries/main_prediction_data_query.txt'

    def __str__(self):
        param_strings = [
            f'project_name = {self.project_name}',
            f'model_version = {self.model_version}',
        ]
        return '\n\t'.join(param_strings)

    def load(self, path, country):
        project_config = load_config(toml.load(path), country)
        self.project_name = project_config['project_name']
        self.model_version = project_config['model_version']
        self.event_query = project_config['event_query_path']
        self.feedback_query = project_config['feedback_query_path']
        self.features_query = project_config['features_query_path']
        self.main_training_data_query = project_config['main_training_data_query_path']
        self.prediction_customers_query_path = project_config['prediction_customers_query_path']
        self.main_prediction_data_query_path = project_config['main_prediction_data_query_path']

    def compile_path(self, partition_params: Dict, file_id, extension):
        partitions = [f'{key}={value}' for key, value in partition_params.items()]
        return str(Path(file_id,
                        f'project={self.project_name}',
                        f'model_version={self.model_version}',
                        *partitions,
                        f'{file_id}.{extension}'))


project_parameters = ProjectParameters()


def load_project_parameters(path, country):
    project_parameters.load(path, country)
