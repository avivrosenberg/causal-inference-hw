import numpy as np
import pandas as pd

"""
This module implements various techniques for estimation of the ATT, i.e.
the average treatment effect on the treated group.
"""


def ipw(y: np.ndarray, t: np.ndarray, propensity: np.ndarray):
    """
    Estimate ATT using inverse-propensity weighting (IPW).
    Assumes a binary treatment, where t=1 is treated and t=0 is control.
    :param y: Outcomes.
    :param t: Treatment.
    :param propensity: Propensity score, i.e. e(x) = P(T=1|X=x).
    """
    assert y.shape == t.shape == propensity.shape

    idx_treat = t == 1

    ipw_weights = propensity / (1 - propensity)
    ipw_weights[idx_treat] = 1.

    att = np.average(y[idx_treat], weights=propensity[idx_treat]) - \
          np.average(y[~idx_treat], weights=propensity[~idx_treat])

    # att = np.sum(y[idx_treat]) / np.sum(propensity[idx_treat]) - \
    #       np.sum(y[~idx_treat] * propensity[~idx_treat]) \
    #       / np.sum(propensity[~idx_treat])
    return att
