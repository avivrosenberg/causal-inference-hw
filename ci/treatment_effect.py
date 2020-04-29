import numpy as np
from numpy import ndarray
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from functools import partial

from .models import interaction_features
from .matching import covariate_matching, propensity_matching

"""
This module implements various techniques for estimation of the causal 
treatment effects ATE and ATT, i.e. the average treatment effect and ATE on 
the treated group.

Assumes binary treatment.

ATE = E[Y^1 - Y^0]
ATT = E[Y^1 - Y^0|T=1]
"""


def ipw(y: ndarray, t: ndarray, propensity: ndarray, att=False):
    """
    Estimate ATE/ATT using inverse-propensity weighting (IPW).
    Assumes a binary treatment, where t=1 is treated and t=0 is control.

    Based on Abdia Y. et. al (2017)
    https://onlinelibrary.wiley.com/doi/pdf/10.1002/bimj.201600094

    @param y: Outcomes.
    @param t: Treatment.
    @param propensity: Propensity score, i.e. e(x) = P(T=1|X=x).
    @param att: Whether to calculate ATT instead of ATE.
    @return: The estimated ATE or ATT.
    """
    assert y.shape == t.shape == propensity.shape

    idx_treat = t == 1

    if att:
        ipw_weights = propensity / (1 - propensity)
        ipw_weights[idx_treat] = 1.
    else:
        ipw_weights = 1 / (1 - propensity)
        ipw_weights[idx_treat] = 1 / propensity[idx_treat]

    att = np.average(y[idx_treat], weights=ipw_weights[idx_treat]) - \
          np.average(y[~idx_treat], weights=ipw_weights[~idx_treat])

    return att


def s_learner(model: BaseEstimator, X: ndarray, y: ndarray,
              t: ndarray, propensity: ndarray = None,
              interaction=False, att=False, transductive=False):
    """
    Estimates the ATE or ATT for a dataset using a trained S-Learner model.
    @param model: The regression model to use for estimation.
    @param X: The covariates shape (N, d)
    @param y: The outcomes, shape (N,l)
    @param t: The treatment assignment, shape (N,)
    @param propensity: Optional propensity scores for doubly-robust estimation.
    @param interaction: Whether to use interaction features. Must be set
    according to the way the model was trained!
    @param att: Whether to calculate ATT instead of ATE.
    @param transductive: Whether to use inductive inference (the default)
    where both the factual outcomes and counterfactual estimates (from the
    model) are used, or transductive inference where only model-estimates
    are used.
    @return: The estimated ATE or ATT.
    """
    assert X.shape[0] == y.shape[0]
    assert y.shape[0] == t.shape[0]
    if propensity is not None:
        assert propensity.shape[0] == y.shape[0]

    # For ATT, select only treatment group
    if att:
        idx_selected = t == 1
    else:
        idx_selected = np.full_like(t, fill_value=True, dtype=np.bool)
    X, y, t = X[idx_selected], y[idx_selected], t[idx_selected]

    # Create interaction features if model was trained with interaction,
    # and set t to one or zero to generate the counterfactual outcome.
    X1 = interaction_features(X, 1, interaction)
    X0 = interaction_features(X, 0, interaction)

    yhat1 = model.predict(X1)
    yhat0 = model.predict(X0)
    if not transductive:
        # Inductive: use factual outcomes
        idx_treated = t == 1
        yhat1[idx_treated] = y[idx_treated]
        yhat0[~idx_treated] = y[~idx_treated]

    if propensity is None:
        return np.average(yhat1 - yhat0, axis=0)

    # If we got propensity scores, use doubly-robust estimator
    p = propensity[idx_selected]
    return doubly_robust(yhat1, yhat0, t, y, p)


def t_learner(treated_model: BaseEstimator,
              control_model: BaseEstimator,
              X: ndarray, y: ndarray,
              t: ndarray, propensity: ndarray = None,
              att=False, transductive=False):
    """
    Estimates the ATE or ATT for a dataset using two trained models,
    in a T-learner approach.
    @param treated_model: The regression model trained on the treated group.
    @param control_model: The regression model trained on the treated group.
    @param X: The covariates shape (N, d)
    @param y: The outcomes, shape (N, l)
    @param t: The treatment assignment, shape (N,)
    @param propensity: Optional propensity scores for doubly-robust estimation.
    @param att: Whether to calculate ATT instead of ATE.
    @param transductive: Whether to use inductive inference (the default)
    where both the factual outcomes and counterfactual estimates (from the
    model) are used, or transductive inference where only model-estimates
    are used.
    @return: The estimated ATE or ATT.
    """

    assert X.shape[0] == y.shape[0]
    assert y.shape[0] == t.shape[0]
    if propensity is not None:
        assert propensity.shape[0] == y.shape[0]

    # For ATT, select only treatment group
    if att:
        idx_selected = t == 1
    else:
        idx_selected = np.full_like(t, fill_value=True, dtype=np.bool)
    X, y, t = X[idx_selected], y[idx_selected], t[idx_selected]

    # Predict outcomes
    yhat1 = treated_model.predict(X)
    yhat0 = control_model.predict(X)

    if not transductive:
        # Inductive: use factual outcomes where available
        idx_treated = t == 1
        yhat1[idx_treated] = y[idx_treated]
        yhat0[~idx_treated] = y[~idx_treated]

    if propensity is None:
        return np.average(yhat1 - yhat0, axis=0)

    # If we got propensity scores, use doubly-robust estimator
    p = propensity[idx_selected]
    return doubly_robust(yhat1, yhat0, t, y, p)


def doubly_robust(
        yhat1: ndarray, yhat0: ndarray, t: ndarray, y: ndarray,
        p: ndarray
) -> ndarray:
    r"""
    ATE/ATT estimation using the "doubly-robust" estimation based on
    propensity scores.

    This method takes the estimation error $\hat{y}_i - y_i$
    into account and weighs it according to the inverse propensity.
    It was shown to produce unbiased estimations of the treatment effect even
    if only one of the models (propensity or outcome regressor) is unbiased.

    Based on: Lunceford, J.K. and Davidian, M.(2004)
    See also: https://www4.stat.ncsu.edu/~davidian/double.pdf

    @param yhat1: Estimated potential outcomes for treated, (N,d).
    @param yhat0: Estimated potential outcomes for control, (N,d)
    @param t: Treatment assignment (N,)
    @param y: Factual outcome (N, d)
    @param p: Propensity scores (N,)
    @return: ATE estimate based on doubly-robust estimator.
    """
    assert yhat1.shape == yhat0.shape == y.shape
    assert y.shape[0] == t.shape[0] == p.shape[0]

    # For multioutcome, use same t and p for all outcomes. Reshape so
    # broadcasting works as we want.
    if y.ndim > 1:
        t = t.reshape(-1, 1)
        p = p.reshape(-1, 1)

    return np.nanmean(
        t * y / p - (t - p) / p * yhat1
        - (1 - t) * y / (1 - p) - (t - p) / (1 - p) * yhat0,
        axis=0
    )


def matching(X: ndarray, y: ndarray, t: ndarray, propensity: ndarray = None,
             method='cosine', k=1, tol=np.inf, att=False):
    """
    Estimates the ATE/ATT for a dataset using a sample-matching approach.

    @param X: The covariates, shape (N, d)
    @param y: The outcomes, shape (N, l)
    @param t: The treatment assignment, shape (N,)
    @param propensity: Optional propensity scores for doubly-robust estimation.
    @param method: See methods of the MatchingEstimator
    @param k: Number of counterfactual neighbors to find and average over.
    @param tol: Tolerance for matching as a fraction of the mean distance
    between pairs with the closest propensity scores. E.g. if tol=0.9 then
    only matching pairs for which the propensity distance is less than 90%
    of the average distance among all pairs will be returned.
    @param att: Whether to calculate ATT instead of ATE.
    @return: Treatment effect, shape (l,)
    """

    assert X.shape[0] == y.shape[0]
    assert y.shape[0] == t.shape[0]
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if propensity is not None:
        assert propensity.shape[0] == y.shape[0]

    idx = np.arange(X.shape[0])
    idx_control, idx_treated = idx[t == 0], idx[t == 1]

    if method.lower() == 'propensity':
        assert propensity is not None
        match_fn = partial(propensity_matching, X=X, t=t, p=propensity,
                           k=k, tol=tol)
    else:
        match_fn = partial(covariate_matching, X=X, t=t, k=k,
                           method=method, tol=tol)

    yhat1, yhat0 = np.copy(y), np.copy(y)
    yhat1[idx_control, :] = np.nan  # yhat1[i] = outcome of sample i with t==1
    yhat0[idx_treated, :] = np.nan  # yhat0[i] = outcome of sample i with t==0

    # Find k-NN matches for treated group in the control group
    _, _, idx_ref_ctrl, idx_query_treated, dists_ref_control = \
        match_fn(match_to=0)

    # Calculate counterfactual outcomes for the control group using
    # weighted-average of matches from treatment group
    # Need a loop because number of matches might not be the same for each
    # query due to the tolerance parameter.
    for j in idx_treated:
        match_idx = idx_query_treated == j
        d_j = dists_ref_control[match_idx]
        if np.sum(d_j) < 1e-9:
            continue
        y_j = y[idx_ref_ctrl[match_idx]]
        y_j_mean = np.average(y_j, axis=0, weights=d_j)
        yhat0[j, :] = y_j_mean

    # For ATT, only use treatment group
    if att:
        yhat1, yhat0 = yhat1[idx_treated], yhat0[idx_treated]
        if propensity is not None:
            y, t, p = y[idx_treated], t[idx_treated], propensity[idx_treated]
            return doubly_robust(yhat1, yhat0, t, y, p)
        else:
            return np.nanmean(yhat1 - yhat0, axis=0)

    # Find k-NN matches for control group in the treated group
    _, _, idx_ref_treated, idx_query_control, dists_ref_treated = \
        match_fn(match_to=1)

    # Calculate counterfactual outcomes for the treated group using
    # weighted-average of matches from the control group
    for j in idx_control:
        match_idx = idx_query_control == j
        d_j = dists_ref_treated[match_idx]
        if np.sum(d_j) < 1e-9:
            continue
        y_j = y[idx_ref_treated[match_idx]]
        y_j_mean = np.average(y_j, axis=0, weights=d_j)
        yhat1[j, :] = y_j_mean

    if propensity is not None:
        return doubly_robust(yhat1, yhat0, t, y, propensity)
    else:
        return np.nanmean(yhat1 - yhat0, axis=0)


def causal_forest(X: ndarray, t: ndarray, cate: ndarray,
                  n_trees=100, test_size=0.5, random_state=None,
                  att=False, **tree_kwargs):
    """
    Estimates the ATT from a dataset using a causal forest approach.
    Requires that the user first estimate the conditional average treatment
    effect (CATE) per sample, e.g. using a T-Learner, matching or any other
    approach.

    This implementation is Based loosely on:
    Wager, S., & Athey, S. (2017). Estimation and inference of heterogeneous
    treatment effects using random forests.
    Journal of the American Statistical Association.
    https://doi.org/10.1080/01621459.2017.1319839

    @param X: Covariates. Shape should be (N, d).
    @param t: The treatment assignment, shape (N,). Assumed to be binary.
    @param cate: Estimated treatment effect per sample, shape (N,).
    @param n_trees: Number of decision trees in the forest.
    @param test_size: Proportion of dataset to use for estimating the ATE/T.
    A different split will be generated for each tree using the same
    proportion.
    @param random_state: Seed for randomization.
    @param tree_kwargs: Use this to pass in extra arguments to the Decision
    Tree model. See the documentation of DecisionTreeRegressor.
    @param att: Whether to calculate ATT instead of ATE.
    @return: The estimated ATE/T.
    """

    if not random_state:
        random_state = np.random.randint(0, 2 ** 30)

    splitter = StratifiedShuffleSplit(n_splits=n_trees,
                                      test_size=test_size,
                                      random_state=random_state)

    att_splits = []
    for idx_train, idx_test in splitter.split(X, t):
        tree = DecisionTreeRegressor(
            random_state=random_state, **tree_kwargs
        )
        tree.fit(X[idx_train], cate[idx_train])

        X_test = X[idx_test]
        if att:
            X_test = X_test[t[idx_test] == 1]

        att_splits.append(np.mean(tree.predict(X_test)))

        random_state += 1

    return np.mean(att_splits)
