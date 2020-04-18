from typing import NamedTuple

import numpy as np
import scipy.stats as stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, \
    RandomizedSearchCV, ShuffleSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

"""
This module implements cross-validation helpers based on the sklearn classes.
"""


class CVConfig(NamedTuple):
    """
    Represents a parameter space for running CV on a specific model.
    """
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




def run_cv_config(cv_config: CVConfig, X: np.ndarray, y: np.ndarray,
                  stratify=None, test_size=0.2, cv_splits=4, scorer=None,
                  n_iter=100, random_state=None):
    """
    Runs cross validation on a single CV configuration (model and parameter
    search space).

    @param cv_config: CVConfig containing a model and a dict of parameters for
    use with RandomizedSearchCV.
    @param X: Features, shape (N, d). Will be split into train/validation/test.
    @param y: Targets, shape (N,).
    @param stratify: Whether to do stratification, both for train/test split
    and also for train/validation splits. If 'True' then y will be used for
    statification. If an ndarray is passed, it must be of shape (N,) and the
    data will be stratified using that.
    @param test_size: Proportion of test set for the train/test split.
    @param cv_splits: Number of CV splits in for K-fold CV.
    @param scorer: Scorer to use. None for classifier default.
    @param n_iter: Number of CV iteration (parameter configs).
    @param random_state: Initial random state.
    @return: A tuple of:
        - CV object containing best-fitted model, can be used for inference
        - Train-set indices
        - Test-set indices
    """
    # Create test-set
    if stratify is not None:
        # Assuming stratify is either True or an ndarray
        y_split = y if stratify is True else stratify
        assert y_split.shape == y.shape
        tt_splitter = StratifiedShuffleSplit(test_size=test_size,
                                             random_state=random_state)
        cv_splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                                      random_state=random_state)
    else:
        y_split = y
        tt_splitter = ShuffleSplit(test_size=test_size,
                                   random_state=random_state)
        cv_splitter = KFold(n_splits=cv_splits, shuffle=True,
                            random_state=random_state)

    # Create a single train-test split
    idx_train, idx_test = next(tt_splitter.split(X, y_split))

    # Create a generator of train-validation splits
    cv = cv_splitter.split(X[idx_train], y_split[idx_train])

    # Add scaling step
    pipeline = Pipeline(steps=[
        ('scale', StandardScaler()),
        ('model', cv_config.model),
    ])

    # Collect parameters into a dict
    rcv_params = {f'model__{k}': v for k, v in cv_config.params.items()}

    # Run the cross-validation
    rcv = RandomizedSearchCV(pipeline, rcv_params, n_iter=n_iter, n_jobs=-1,
                             cv=cv, scoring=scorer, refit=True,
                             random_state=random_state)
    rcv.fit(X[idx_train], y[idx_train])
    return rcv, idx_train, idx_test
