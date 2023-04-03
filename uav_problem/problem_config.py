PROBLEM_CONFIG = None


def set_config(config):
    global PROBLEM_CONFIG
    PROBLEM_CONFIG = config


def get_config(assert_not_none=True):
    assert assert_not_none and PROBLEM_CONFIG is not None, "Problem config not set"
    return PROBLEM_CONFIG
