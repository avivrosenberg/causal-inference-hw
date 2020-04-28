import numpy as np
from sklearn.metrics import make_scorer, r2_score

from .cv import CVConfig, run_cv_config


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
