from typing import Optional, List
import pandas as pd
import statsmodels
import statsmodels.api as sm


def ols_groupby(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None) -> \
        List[statsmodels.regression.linear_model.RegressionResultsWrapper]:
    """
    Does ols, separating by hue (as used by seaborn)

    Parameters
    ----------
    df
    x
    y
    hue

    Returns
    -------

    """

    if hue is not None:
        hue_vals = df[hue].unique()
    else:
        hue_vals = [None]

    all_results = []
    for val in hue_vals:
        x_vec, y_vec = df[x], df[y]
        if hue is not None:
            ind = df[hue] == val
            x_vec = x_vec[ind]
            y_vec = y_vec[ind]

        x_vec = sm.add_constant(x_vec)
        results = sm.OLS(y_vec, x_vec, missing='drop').fit()

        all_results.append(results)

    return all_results
