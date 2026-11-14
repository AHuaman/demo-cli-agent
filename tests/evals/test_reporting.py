import pandas as pd

from tests.evals.reporting import judge_pass_rates


def test_judge_pass_rates_averages_yes_no():
    df = pd.DataFrame({"safety_judge/value": ["yes", "no", "yes", "yes"]})
    assert judge_pass_rates(df, ["safety_judge"]) == {"safety_judge/mean": 0.75}


def test_judge_pass_rates_skips_errored_rows():
    # A row whose judge call failed has no value; it must not count as a fail.
    df = pd.DataFrame({"safety_judge/value": ["yes", None, "no"]})
    assert judge_pass_rates(df, ["safety_judge"]) == {"safety_judge/mean": 0.5}


def test_judge_pass_rates_omits_judge_without_values():
    df = pd.DataFrame({"a/value": ["yes"]})
    assert judge_pass_rates(df, ["a", "missing"]) == {"a/mean": 1.0}
