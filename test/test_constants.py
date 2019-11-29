from voucher_opt.constants import Environment


def test_get_env():
    env = Environment.get_env('production')
    assert env == Environment.PRODUCTION
    env = Environment.get_env('staging')
    assert env == Environment.STAGING
    env = Environment.get_env('development')
    assert env == Environment.DEVELOPMENT
