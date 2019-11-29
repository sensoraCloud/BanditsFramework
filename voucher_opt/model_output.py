import numpy as np
import pandas as pd

from voucher_opt.config.global_parameters import GlobalParameters
from voucher_opt.config.project_parameters import project_parameters
from voucher_opt.constants import ACTION_CODE, LOGPROB, COUNTRY, SEGMENT, EXPLORATION, CONTROL, ACTION_GENERATED_DATE, \
    GIVER_ID, MODEL_ID, ACTION_DESC, MODEL_VERSION, PROJECT
from voucher_opt.logger import log
from voucher_opt.utils import print_header, print_footer

VALIDATION_EPSILON = 0.005

FINAL_MODEL_OUTPUT_COLUMNS = [ACTION_CODE, ACTION_DESC, ACTION_GENERATED_DATE, CONTROL, COUNTRY, EXPLORATION, GIVER_ID,
                              LOGPROB, MODEL_ID, MODEL_VERSION, PROJECT, SEGMENT]


class ModelOutput:
    def __init__(self, df):
        self._output_df = df

    @staticmethod
    def create_from_transformed_data(predictions_df, other_groups_df):
        output_df = pd.concat([predictions_df, other_groups_df])
        return ModelOutput(output_df)

    @staticmethod
    def _create_from_file(path):
        return ModelOutput(pd.read_csv(path))

    def prepare_output(self, country, model_id, prediction_date, action_desc):
        self._output_df[PROJECT] = project_parameters.project_name
        self._output_df[MODEL_VERSION] = project_parameters.model_version
        self._output_df[COUNTRY] = country
        self._output_df[MODEL_ID] = model_id
        self._output_df[ACTION_GENERATED_DATE] = prediction_date
        self._output_df[ACTION_DESC] = action_desc
        self._output_df = self._output_df[FINAL_MODEL_OUTPUT_COLUMNS]
        self._output_df = self._output_df.dropna(subset=[GIVER_ID])
        self._set_types()

        self._output_df.sort_index(axis=1, inplace=True)

        log.debug(f'model_output.columns = {self._output_df.columns}')

    def validate(self, global_parameters: GlobalParameters):
        log.info('Validating model output...')
        self._validate_group(CONTROL, 0, 'not_control', 1 - global_parameters.control - global_parameters.other_action)
        self._validate_group(CONTROL, 1, 'control', global_parameters.control)
        self._validate_group(CONTROL, 2, 'other_action', global_parameters.other_action)
        explore_percentage = \
            (1 - global_parameters.control - global_parameters.other_action) * global_parameters.exploration
        self._validate_group(EXPLORATION, 0, 'not_explore', 1 - explore_percentage)
        self._validate_group(EXPLORATION, 1, 'explore', explore_percentage)
        log.info('Done!')
        return self._output_df

    def log_stats(self):
        model_output = self._output_df
        print_header('OUTPUT')
        log.info(f'Control value distribution:\n'
                 f'{model_output.control.value_counts().sort_index()}\n')
        log.info(f'Exploration value distribution:\n'
                 f'{model_output.exploration.value_counts().sort_index()}\n')

        logprob_value_counts = model_output.logprob \
            .apply(round, ndigits=3) \
            .value_counts() \
            .nlargest(10) \
            .sort_index(ascending=False)
        log.info(
            f'''Logprob value distribution:\n{logprob_value_counts}\n'''
        )

        log.info(f'Action distribution for all customers:\n'
                 f'{model_output.action_code.value_counts().sort_index()}\n')

        log.info(
            f'Action distribution for exploration group:\n'
            f'{model_output[model_output["exploration"] == 1].action_code.value_counts().sort_index()}\n'
        )

        model_action_distribution = self.model_action_distribution()
        log.info(
            f'''Action distribution for experimental group:\n{model_action_distribution.sort_index()}\n''')

        print_footer()

    def model_action_distribution(self):
        return self._output_df[(self._output_df[CONTROL] == 0) &
                               (self._output_df[EXPLORATION] == 0)].action_code.value_counts()

    def _set_types(self):
        self._output_df = self._output_df.astype({
            ACTION_CODE: np.float64,
            ACTION_DESC: str,
            ACTION_GENERATED_DATE: str,
            CONTROL: np.int64,
            COUNTRY: str,
            EXPLORATION: np.int64,
            GIVER_ID: np.int64,
            LOGPROB: np.float64,
            MODEL_ID: str,
            MODEL_VERSION: str,
            PROJECT: str,
            SEGMENT: str
        })

    def _validate_group(self, group_partition, group_id, group_name, correct_value):
        value_counts = self._output_df[group_partition].value_counts()
        total = value_counts.sum()
        count = value_counts.get(group_id, default=0)
        percentage = count / total
        error_message = f'{group_name} group should be {correct_value:.2%}, but is {percentage:.2%}'
        assert correct_value - VALIDATION_EPSILON < percentage < correct_value + VALIDATION_EPSILON, error_message
