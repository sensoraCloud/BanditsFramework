from enum import Enum

from voucher_opt.file_handling.file_utils import read_query


class FeatureType(Enum):
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    BINARY = 'binary'


class Feature:
    def __init__(self, original_name: str, short: str, f_type: FeatureType, query_name: str):
        self.original_name = original_name
        self.short_name = short
        self.f_type = f_type
        self.query_name = query_name

    def __eq__(self, other):
        if isinstance(other, Feature):
            return self.original_name == other.original_name and \
                   self.f_type == other.f_type and \
                   self.query_name == other.query_name
        return False

    def __str__(self) -> str:
        return f'{self.original_name} ({self.f_type.name})'

    def __repr__(self):
        return self.__str__()

    # Note: When adding more features, there is a high risk of collisions among the short names.
    @staticmethod
    def create(name, feature_type, query_name):
        return Feature(name, short_name(name), feature_type, query_name)


def short_name(name):
    return ''.join([part[0] for part in name.split('_')])


DEFAULT_FEATURES = (
    Feature.create('activation_channel', FeatureType.CATEGORICAL, 'mtc_activation_info'),
    Feature.create('uses_iphone', FeatureType.BINARY, 'devices'),
    Feature.create('uses_mac', FeatureType.BINARY, 'devices'),
    Feature.create('uses_linux', FeatureType.BINARY, 'devices'),
    Feature.create('app_installed', FeatureType.BINARY, 'devices'),
    Feature.create('gir_sent', FeatureType.NUMERICAL, 'gir_sent_info'),
    Feature.create('gir_converted', FeatureType.NUMERICAL, 'gir_sent_redeemed_info'),
    Feature.create('helloshare_sent', FeatureType.NUMERICAL, 'helloshare_sent_info'),
    Feature.create('helloshare_sent_converted', FeatureType.NUMERICAL, 'helloshare_sent_converted'),
    Feature.create('week_swap_rate', FeatureType.NUMERICAL, 'meal_swaps_info'),
    Feature.create('num_errors', FeatureType.NUMERICAL, 'customer_care_interact_info'),
    Feature.create('count_premium', FeatureType.NUMERICAL, 'up_selling_info'),
    Feature.create('boxes_shipped', FeatureType.NUMERICAL, 'loyalty_info'),
    Feature.create('box_rate', FeatureType.NUMERICAL, 'loyalty_info'),
    Feature.create('loyalty_score', FeatureType.NUMERICAL, 'loyalty_info'),
    Feature.create('tot_reactivations', FeatureType.NUMERICAL, 'reactivation'),
    Feature.create('perc_paused', FeatureType.NUMERICAL, 'past_paused_rate')
)

countries = ['AU', 'BE', 'CA', 'DE', 'GB', 'NL', 'US']

COUNTRY_TO_FEATURES = {
    country: DEFAULT_FEATURES for country in countries
}

COUNTRY_TO_FEATURES['US'] += (
    Feature.create('act_int_gourmet_cooking_hh', FeatureType.NUMERICAL, 'experian'),
    Feature.create('act_int_eats_at_family_restaurants_hh', FeatureType.NUMERICAL, 'experian'),
    Feature.create('act_int_eats_at_fast_food_restaurants_hh', FeatureType.NUMERICAL, 'experian'),
    Feature.create('person_1_gender_idl', FeatureType.CATEGORICAL, 'experian'),
    Feature.create('estimated_household_income_range_code_v6_hh', FeatureType.CATEGORICAL, 'experian'),
    Feature.create('mosaic_global_household_hh', FeatureType.CATEGORICAL, 'experian'),
    Feature.create('number_of_persons_in_living_unit_hh', FeatureType.NUMERICAL, 'experian'),
    Feature.create('cape_inc_hh_median_family_household_income_geo', FeatureType.NUMERICAL, 'experian'),
    Feature.create('cape_age_pop_median_age_geo', FeatureType.NUMERICAL, 'experian'),
    Feature.create('household_composition_hh', FeatureType.CATEGORICAL, 'experian'),
    Feature.create('cape_educ_ispsa_decile_geo', FeatureType.NUMERICAL, 'experian'),
    Feature.create('census_rural_urban_county_size_code_geo', FeatureType.CATEGORICAL, 'experian')
)


class FeatureSet:
    def __init__(self, features):
        self._features = features

    @staticmethod
    def create_for_country(country):
        return FeatureSet(COUNTRY_TO_FEATURES[country])

    def all_features(self) -> [Feature]:
        return self._features

    def experian_features(self) -> [Feature]:
        return [feature for feature in self._features if feature.query_name == 'experian']

    def has_experian_features(self):
        return len(self.experian_features()) > 0

    def numerical_features(self) -> [Feature]:
        return [f for f in self._features if f.f_type == FeatureType.NUMERICAL]

    def categorical_features(self) -> [Feature]:
        return [f for f in self._features if f.f_type == FeatureType.CATEGORICAL]


def shorten_feature_names(df, feature_columns):
    rename_dict = {col: short_name(col) for col in feature_columns}
    df = df.rename(index=str, columns=rename_dict)
    return df


# TODO: Describe what this method does
# TODO: Make it clear that this generates one big query to be saved and used later
def generate_features_sub_queries(features):
    feature_query_names = sub_query_names(features)
    sub_query_snippets = features_sub_query_snippets(feature_query_names)
    features_sub_queries_str = ',\n'.join(sub_query_snippets)

    features_column_str = ',\n\t\t'.join(
        [f'{f.query_name}.{f.original_name} AS {f.original_name}' for f in features])

    sub_query_join_str = ''.join([f'''
    LEFT JOIN
        {query_name}
        ON {query_name}.customer_id = customers.customer_id''' for query_name in feature_query_names])

    generated_sub_queries = f'''{features_sub_queries_str},
features AS (
    SELECT
        customers.customer_id AS giver_id,
        {features_column_str}
    FROM
        customers {sub_query_join_str}
)'''

    return generated_sub_queries


def sub_query_names(features):
    return set(feature.query_name for feature in features)


def features_sub_query_snippets(sub_query_ids):
    sub_queries = []
    for query_name in sub_query_ids:
        query_file = f'queries/feature_queries/{query_name}.txt'
        sub_queries.append(read_query(query_file))
    return sub_queries


def feature_names(features) -> [str]:
    return [feature.original_name for feature in features]
