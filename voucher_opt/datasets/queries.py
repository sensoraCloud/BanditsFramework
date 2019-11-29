import hfds

from voucher_opt.config.project_parameters import project_parameters
from voucher_opt.features.feature_definitions import generate_features_sub_queries


# TODO: Add comments about what happens here, especially what it produces
# TODO: Docstrings and SQL comments
def compile_dataset_query(model_version, country, prediction_date, feedback_weeks, event_table_identifier, features):
    event_query = read_query(project_parameters.event_query, country=country, model_version=model_version,
                             current_hf_running_week=_get_current_running_hf_week(prediction_date),
                             feedback_weeks=feedback_weeks, table_identifier=event_table_identifier)
    feedback_query = read_query(project_parameters.feedback_query, feedback_weeks=feedback_weeks)
    features_query = generate_features_sub_queries(features)
    main_training_data_query = read_query(project_parameters.main_training_data_query)
    complete_dataset_query = ', '.join(
        (event_query, feedback_query, features_query, main_training_data_query)) + ' SELECT * FROM dataset'
    return complete_dataset_query


def compile_prediction_data_query(country, elaboration_date_str, features):
    prediction_customers_query = read_query(project_parameters.prediction_customers_query_path, country=country,
                                            ref_date=elaboration_date_str)
    features_query = generate_features_sub_queries(features)
    main_prediction_data_query = read_query(project_parameters.main_prediction_data_query_path)
    complete_prediction_data_query = ', '.join(
        (prediction_customers_query, features_query, main_prediction_data_query)) + ' SELECT * FROM prediction_data'
    return complete_prediction_data_query


def _get_current_running_hf_week(prediction_date):
    current_hf_week = hfds.dt.datetime_to_hf_week(prediction_date)
    running_week_query = f'''
        SELECT DISTINCT
            hellofresh_running_week
        FROM
            dimensions.date_dimension
        WHERE
            hellofresh_week = "{current_hf_week}"
        '''
    current_hf_running_week = hfds.db.run_dwh_query(running_week_query).values[0][0]
    return current_hf_running_week


def read_query(query_path, **query_params):
    with open(query_path) as query_file:
        query = query_file.read().format(**query_params)
    return query
