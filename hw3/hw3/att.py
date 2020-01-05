import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.tree import DecisionTreeRegressor

from .cv import CVConfig, run_cv_config

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


def interaction_features(X, t, interaction=True):
    """
    Creates interaction features for an s-learner model.
    @param X: Covariates. Shape should be (N, d)
    @param t: Treatment. Can be scalar, in which case same treatment will
    be used for all samples.
    @param interaction: Whether to create interaction features between X and t.
    Can be True, False or None. If True, both X*t and t will be added as
    covariates. If False, only t will be added. If None, nothing will be
    added and X will be returned as-is.
    @return: Features matrix of shape (N, d+1) if interaction=False,
    or of shape (N, 2d+1) if interaction=True. The +1 is t and the extra d
    are the interactions.
    """
    if interaction is None:
        return X

    if isinstance(t, (int, float)):
        t = np.full(shape=(X.shape[0], 1), fill_value=t)

    t = t.reshape(-1, 1)
    if interaction:
        X = np.hstack((X, X * t, t))
    else:
        X = np.hstack((X, t))

    return X


def fit_slearner_cv(cv_cfg: CVConfig, X: np.ndarray, y: np.ndarray,
                    t: np.ndarray, interaction=False, **cv_args):
    """
    Fits an s-learner using randomized-search cross-validation with
    stratification on the treatment assignment.
    @param cv_cfg: CVConfig containing a model and a dict of parameters for
    use with RandomizedSearchCV.
    @param X: Covariates. Will be split into train/validation/test.
    @param y: Outcomes.
    @param t: Treatment assignment
    @param interaction: Whether to create interaction features between
    covariates and treatment.
    @return: A tuple of:
        - CV object containing best-fitted model, can be used for inference
        - Train-set R^2 score
        - Test-set R^2 score
    """
    X = interaction_features(X, t, interaction)

    scorer = make_scorer(r2_score, greater_is_better=True, needs_proba=False)

    rcv, idx_train, idx_test = run_cv_config(
        cv_cfg, X, y, stratify=t, scorer=scorer, **cv_args
    )

    # Infer scores on best model
    train_score = rcv.score(X[idx_train], y[idx_train])
    test_score = rcv.score(X[idx_test], y[idx_test])

    return rcv, train_score, test_score


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
    X1 = interaction_features(X, 1, interaction)
    X0 = interaction_features(X, 0, interaction)

    yhat1 = model.predict(X1)
    yhat0 = model.predict(X0)
    if propensity is None:
        return np.average(yhat1 - yhat0)

    # If we got propensity scores, use doubly-robust estimator
    e = propensity[idx_treat]
    return doubly_robust(yhat1, yhat0, t, y, e)


def t_learner(treated_model: BaseEstimator,
              control_model: BaseEstimator,
              X: np.ndarray, y: np.ndarray,
              t: np.ndarray, propensity: np.ndarray = None):
    """
    Estimates the ATT for a dataset using two trained models, in a T-learner
    approach.
    @param treated_model: The regression model trained on the treated group.
    @param control_model: The regression model trained on the treated group.
    @param X: The covariates shape (N, d)
    @param y: The outcomes, shape (N,)
    @param t: The treatment assignment, shape (N,)
    @param propensity: Optional propensity scores for doubly-robust estimation.
    @return: The estimated ATT.
    """

    assert X.shape[0] == y.shape[0]
    assert y.shape == t.shape
    if propensity is not None:
        assert propensity.shape == y.shape

    # Select only treatment group
    idx_treat = t == 1
    X, y, t = X[idx_treat], y[idx_treat], t[idx_treat]

    # Predict factual outcomes
    yhat1 = treated_model.predict(X)

    # Predict counterfactual outcomes
    yhat0 = control_model.predict(X)

    if propensity is None:
        return np.average(yhat1 - yhat0)

    # If we got propensity scores, use doubly-robust estimator
    e = propensity[idx_treat]
    return doubly_robust(yhat1, yhat0, t, y, e)


def doubly_robust(yhat1, yhat0, t, y, e):
    assert yhat1.shape == yhat0.shape == t.shape == y.shape == e.shape

    # Formula based on https://www4.stat.ncsu.edu/~davidian/double.pdf
    return np.average(
        t * y / e - (t - e) / e * yhat1
        - (1 - t) * y / (1 - e) - (t - e) / (1 - e) * yhat0
    )


def matching(y, idx_treat_m, idx_ctrl_m):
    """
    Estimates the ATT for a dataset given matched samples.
    @param y: Outcome values, of shape (N,).
    @param idx_treat_m: Indices of matched samples that belong to the
    treatment group, of shape (M,) where M <= N.
    @param idx_ctrl_m: Indices of matched samples that belong to the
    control group, of shape (M,) where M <= N.
    @return: The estimated ATT.
    """
    return np.mean(y[idx_treat_m] - y[idx_ctrl_m])


def causal_forest(X: np.ndarray, t: np.ndarray, cate: np.ndarray,
                  n_trees=100, test_size=0.5, random_state=None,
                  **tree_kwargs):
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
    @param test_size: Proportion of dataset to use for estimating the ATT.
    A different split will be generated for each tree using the same
    proportion.
    @param random_state: Seed for randomization.
    @param tree_kwargs: Use this to pass in extra arguments to the Decision
    Tree model. See the documentation of DecisionTreeRegressor.
    @return: The estimated ATT.
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

        X_test_treated = X[idx_test][t[idx_test] == 1]
        att_splits.append(np.mean(
            tree.predict(X_test_treated)
        ))

        random_state += 1

    return np.mean(att_splits)
