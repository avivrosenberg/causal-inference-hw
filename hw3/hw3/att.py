import numpy as np
from sklearn.base import BaseEstimator

from .cv import slearner_features

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

    att = np.average(y[idx_treat], weights=ipw_weights[idx_treat]) - \
          np.average(y[~idx_treat], weights=ipw_weights[~idx_treat])

    return att


def s_learner(model: BaseEstimator, X: np.ndarray, y: np.ndarray,
              t: np.ndarray, propensity: np.ndarray = None, interaction=False):
    """
    Estimates the ATT for a dataset using a trained S-Learner model.
    @param model: The regression model to use for estimation.
    @param X: The covariates shape (N, d)
    @param y: The outcomes, shape (N,)
    @param t: The treatment assignment, shape (N,)
    @param propensity: Optional propensity scores for doubly-robust estimation.
    @param interaction: Whether to use interaction features. Must be set
    according to the way the model was trained.
    @return: The estimated ATT.
    """
    assert X.shape[0] == y.shape[0]
    assert y.shape == t.shape
    if propensity is not None:
        assert propensity.shape == y.shape

    # Select only treatment group
    idx_treat = t == 1
    X, y, t = X[idx_treat], y[idx_treat], t[idx_treat]

    # Create interaction features if model was trained with interaction,
    # and set t to one or zero to generate the counterfactual outcome.
    X1 = slearner_features(X, 1, interaction)
    X0 = slearner_features(X, 0, interaction)

    yhat1 = model.predict(X1)
    yhat0 = model.predict(X0)
    if propensity is None:
        return np.average(yhat1 - yhat0)

    # If we got propensity scores, use doubly-robust estimator
    e = propensity[idx_treat]

    # Formula based on https://www4.stat.ncsu.edu/~davidian/double.pdf
    return np.average(
        t * y / e - (t - e) / e * yhat1
        - (1 - t) * y / (1 - e) - (t - e) / (1 - e) * yhat0
    )
