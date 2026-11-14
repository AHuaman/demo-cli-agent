import pandas as pd

# MLflow 3.8.1 builds every `make_judge` judge with `aggregations=[]`, so
# `evaluate()` never turns judge feedback into run metrics. Evals compute each
# judge's pass rate from the per-row results instead.
_VERDICTS = {"yes": 1.0, "no": 0.0}


def judge_pass_rates(result_df: pd.DataFrame, judge_names: list[str]) -> dict[str, float]:
    """Mean of yes(1)/no(0) per judge, as `<judge>/mean`. Rows without a
    verdict (judge call errored) are left out rather than counted as a fail."""
    rates = {}
    for name in judge_names:
        column = f"{name}/value"
        if column not in result_df:
            continue
        scores = [_VERDICTS[v] for v in result_df[column] if v in _VERDICTS]
        if scores:
            rates[f"{name}/mean"] = sum(scores) / len(scores)
    return rates
