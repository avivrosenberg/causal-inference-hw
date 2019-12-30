from typing import NamedTuple

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score, make_scorer
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, \
    RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""
This module implements cross-validation for fitting various types of models.
"""


class CVConfig(NamedTuple):
    model: BaseEstimator
    params: dict


class MLPHiddenLayerSampler:
    """
    Samples hidden layers and their dimentions for use in cross validation
    of an MLP model.
    """

    def __init__(self, max_n_layers=10, max_h_dim=100):
        self.n_layers = stats.randint(1, max_n_layers + 1)
        self.h_dim = stats.randint(1, max_h_dim + 1)

    def rvs(self, random_state=None):
        return tuple(
            self.h_dim.rvs(random_state=random_state)
            for i in range(self.n_layers.rvs(random_state=random_state))
        )


class LogSpaceSampler:
    """
    Samples from a log-space between 10^x to 10^y.
    """

    def __init__(self, min_power=-2, max_power=2):
        assert max_power > min_power
        self.sampler = stats.uniform(min_power, max_power - min_power)

    def rvs(self, random_state=None):
        return 10 ** self.sampler.rvs(random_state=random_state)


def slearner_features(X, t, interaction=True):
    """
    Creates input features for an s-learner model.
    @param X: Covariates. Shape should be (N, d)
    @param t: Treatment. Can be scalar, in which case same treatment will
    be used for all samples.
    @param interaction: Whether to create interaction features between X and t.
    @return: Features matrix of shape (N, d+1) if interaction=False,
    or of shape (N, 2d+1) if interaction=True. The +1 is t and the extra d
    are the interactions.
    """
    if isinstance(t, (int, float)):
        t = np.full(shape=(X.shape[0], 1), fill_value=t)

    t = t.reshape(-1, 1)
    if interaction:
        X = np.hstack((X, X * t, t))
    else:
        X = np.hstack((X, t))

    return X


def fit_slearner_cv(cv_cfg: CVConfig, X: np.ndarray, y: np.ndarray,
                    t: np.ndarray, interaction=False, test_size=0.2,
                    train_splits=4, n_iter=100, random_state=None):
    """
    Fits an s-learner using randomized-search cross-validation.
    @param cv_cfg: CVConfig containing a model and a dict of parameters for
    use with RandomizedSearchCV.
    @param X: Covariates. Will be split into train/validation/test.
    @param y: Outcomes.
    @param t: Treatment assignment
    @param interaction: Whether to create interaction features between
    covariates and treatment.
    @param test_size:
    @param train_splits:
    @param n_iter:
    @param random_state:
    @return: A tuple of:
        - CV object containing best-fitted model, can be used for inference
        - Train-set R^2 score
        - Test-set R^2 score

    """
    X = slearner_features(X, t, interaction)

    # Create test-set, stratify on t
    split = StratifiedShuffleSplit(test_size=test_size,
                                   random_state=random_state)
    idx_train, idx_test = next(split.split(X, t))

    # Create validation sets, stratify on t
    split = StratifiedKFold(n_splits=4, shuffle=True,
                            random_state=random_state)
    cv = split.split(X[idx_train], t[idx_train])
    scorer = make_scorer(r2_score, greater_is_better=True, needs_proba=False)
    pipeline = Pipeline(steps=[
        ('scale', StandardScaler()),
        ('model', cv_cfg.model),
    ])
    rcv_params = {f'model__{k}': v for k, v in cv_cfg.params.items()}
    rcv = RandomizedSearchCV(pipeline, rcv_params, n_iter=n_iter, n_jobs=-1,
                             cv=cv, scoring=scorer, iid=False, refit=True,
                             random_state=random_state)

    rcv.fit(X[idx_train], y[idx_train])
    train_score = rcv.score(X[idx_train], y[idx_train])
    test_score = rcv.score(X[idx_test], y[idx_test])

    return rcv, train_score, test_score
