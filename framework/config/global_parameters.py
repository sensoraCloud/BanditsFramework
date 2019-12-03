from voucher_opt import constants
from voucher_opt.utils import validate_bounds, validate_type


class GlobalParameters():
    def __init__(self, control, other_action, exploration, actions, default_action, action_desc, feedback_weeks):
        self.control = control
        self.other_action = other_action
        self.exploration = exploration
        self.actions = actions
        self.default_action = default_action
        self.action_desc = action_desc
        self.feedback_weeks = feedback_weeks

    def __str__(self):
        param_strings = [f'control =  {self.control}',
                         f'other_action = {self.other_action}',
                         f'exploration = {self.exploration}',
                         f'actions = {self.actions}',
                         f'default_action = {self.default_action}',
                         f'action_desc = {self.action_desc}',
                         f'feedback_weeks = {self.feedback_weeks}']
        return '\n\t'.join(param_strings)

    @staticmethod
    def create_from_run_config(run_config):
        return GlobalParameters(run_config[constants.CONTROL_KEY],
                                run_config[constants.OTHER_ACTION_KEY],
                                run_config[constants.EXPLORATION_KEY],
                                run_config[constants.ACTIONS_KEY],
                                run_config[constants.DEFAULT_ACTION_KEY],
                                run_config[constants.ACTION_DESC_KEY],
                                run_config[constants.FEEDBACK_WEEKS_KEY])

    @property
    def experimental_group(self):
        no_control_perc = 1.0 - self.control - self.other_action
        return round(no_control_perc - (no_control_perc * self.exploration), 3)

    def validate(self):
        validate_bounds('exploration', self.exploration, 0.0, 1.0)
        validate_bounds('control', self.control, 0.0, 1.0)
        validate_bounds('other_action', self.other_action, 0.0, 1.0)
        validate_bounds('control + other_action', self.control + self.other_action, 0.0, 1.0)
        assert not (self.control + self.other_action >= 1.0 and self.exploration > 0), \
            'Exploration can not be > 0 if control is 1.0.'
        assert self.default_action in self.actions, \
            f'Default action "{self.default_action}" is not in the list of available actions: {self.actions}'
        validate_bounds('feedback_weeks', self.feedback_weeks, 0, 52)
        validate_type('feedback_weeks', self.feedback_weeks, int)
