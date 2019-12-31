import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from .cv import CVConfig, run_cv_config


def fit_propensity_cv(cv_cfg: CVConfig, X: np.ndarray, t: np.ndarray,
                      plot_args=None, **cv_args):
    scorer = None
    cv_model, idx_train, idx_test = run_cv_config(
        cv_cfg, X, t, stratify=True, scorer=scorer, **cv_args
    )

    plot_args = plot_args if plot_args else dict()
    best_est, best_score = calibrate_classifier(
        cv_model.best_estimator_, X, t, idx_calib=idx_test, **plot_args
    )

    return best_est, cv_model.best_params_


def calibrate_classifier(estimator, X, y, idx_calib, **plot_kw):
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
         CalibratedClassifierCV(estimator, cv='prefit', method='isotonic')
         .fit(X[idx_calib], y[idx_calib])
         ),
        (est_name + ' platt',
         CalibratedClassifierCV(estimator, cv='prefit', method='sigmoid')
         .fit(X[idx_calib], y[idx_calib])
         ),
    ]

    scores, aurocs = [], []
    for name, est in estimators:
        proba = est.predict_proba(X)[:, 1]
        scores.append(brier_score_loss(y, proba, pos_label=y.max()))
        aurocs.append(roc_auc_score(y, proba))

        # Plot calibration curve
        if plot_kw:
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y, proba, n_bins=15)
            ax.plot(mean_predicted_value, fraction_of_positives, "s-",
                    label=f"{name} (a={aurocs[-1]:.3f} b={scores[-1]:.3f})")

    if plot_kw:
        ax.legend(loc='lower right')
        ax.grid(True)

    best_est = estimators[np.argmax(scores)][1]
    return best_est, np.max(scores)


def common_support(t: np.ndarray, propensity: np.ndarray,
                   min_thresh=1e-5, max_thresh=1-1e-5):
    """
    Returns the common support between Treatment and Control groups in terms
    of propensity score overlap.
    Also allows rejecting samples with a propensity lower/higher than some
    threshold.

    @param t: Treatment assignments (binary), shape (N,)
    @param prop: Propensity scores, shape (N,)
    @param min_thresh: Minimal propensity score to keep. Samples with a
    lower score will be rejected. Zero has no effect.
    @param max_thresh: Maximal propensity score to keep. Samples with a
    higher score will be rejected. One has no effect.
    @return: Sample indices of the common support.
    """
    assert t.ndim == 1
    assert t.shape == propensity.shape
    assert min_thresh >= 0
    assert max_thresh <= 1

    idx_treat = t == 1

    # Lower-bound of common support: maximum between minimal propensity of
    # each group and the minimal-value threshold
    lower_bound = max(np.min(propensity[idx_treat]),
                      np.min(propensity[~idx_treat]),
                      min_thresh)

    # Upper-bound of common support: minimum between maximal propensity of
    # each group and the maximal-value threshold
    upper_bound = min(np.max(propensity[idx_treat]),
                      np.max(propensity[~idx_treat]),
                      max_thresh)

    # Return sample indices within the common support
    return (propensity >= lower_bound) & (propensity <= upper_bound)
