import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold

SEED = 42


def calibrate_classifier(estimator, X, y, cv_splits=5, **plot_kw):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(cv.split(X, y))

    if plot_kw:
        ax = plot_kw.get('ax')
        if not ax:
            figsize = plot_kw.get('figsize', (8, 6))
            _, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot([0, 1], [0, 1], "k:", label="Ideal calibration")
        ax.set_ylabel("Fraction of positives")
        ax.set_xlabel("Predicted probability")
        ax.set_ylim([-0.05, 1.05])

    est_name = plot_kw.get('name', estimator.__class__.__name__)
    estimators = [
        (est_name, estimator),
        (est_name + ' isotonic',
         CalibratedClassifierCV(estimator, cv=cv, method='isotonic')),
        (est_name + ' platt',
         CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')),
    ]

    scores, aurocs = [], []
    for name, est in estimators:
        est.fit(X[train_idx], y[train_idx])
        proba = est.predict_proba(X[valid_idx])[:, 1]
        scores.append(brier_score_loss(y[valid_idx], proba, pos_label=y.max()))
        aurocs.append(roc_auc_score(y[valid_idx], proba))

        # Plot calibration curve
        if plot_kw:
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y[valid_idx], proba, n_bins=15)
            ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f"{name} (a={aurocs[-1]:.3f} b={scores[-1]:.3f})")

    if plot_kw:
        ax.legend(loc='lower right')
        ax.grid(True)

    best_est = estimators[np.argmax(scores)]
    return best_est[1], np.max(scores)


SUPPORTED_METHODS = {
    'logistic': LogisticRegressionCV,
    'gbm': GradientBoostingClassifier,
}


def estimate_propensity(X: np.ndarray, t: np.ndarray, method='logistic',
                        plot_args=None, **est_kw):
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unknown method {method}. Must be in "
                         f"{SUPPORTED_METHODS.keys()}")

    if method == 'logistic':
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scorer = make_scorer(brier_score_loss, greater_is_better=False,
                             needs_proba=True)
        default_est_kw = dict(penalty='l1', Cs=20, solver='liblinear', cv=cv,
                              scoring=scorer)
    elif method == 'gbm':
        default_est_kw = dict(learning_rate=0.01, max_depth=2)
    else:
        raise ValueError(f"Unknown method {method}")

    # Merge est_kw with defaults so that est_kw has precedence
    for k, v in default_est_kw.items():
        est_kw.setdefault(k, v)

    estimator = SUPPORTED_METHODS[method](**est_kw)

    plot_args = plot_args if plot_args else dict()
    calibrated_estimator, _ = calibrate_classifier(
        estimator, X, t, **plot_args
    )

    propensity = calibrated_estimator.predict_proba(X)[:, 1]
    return propensity


def common_support(t: np.ndarray, propensity: np.ndarray):
    """

    @param t:
    @param prop:
    @return: Sample indices of the common support.
    """
    assert t.ndim == 1
    assert t.shape == propensity.shape

    idx_treat = t == 1

    # Lower-bound of common support: maximum between minimal propensity of
    # each group
    lower_bound = max(np.min(propensity[idx_treat]),
                      np.min(propensity[~idx_treat]))

    # Upper-bound of common support: minimum between maximal propensity of
    # each group
    upper_bound = min(np.max(propensity[idx_treat]),
                      np.max(propensity[~idx_treat]))

    # Return sample indices within the common support
    return (propensity >= lower_bound) & (propensity <= upper_bound)
