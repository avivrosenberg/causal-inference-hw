import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import brier_score_loss, make_scorer
from sklearn.model_selection import StratifiedKFold

SEED = 42


def calibrate_classifier(estimator, X, y, cv_splits=5, plot=False):
    est_name = estimator.__class__.__name__
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    train_idx, valid_idx = next(cv.split(X, y))

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.plot([0, 1], [0, 1], "k:", label="Ideal calibration")
        ax.set_ylabel("Fraction of positives")
        ax.set_ylim([-0.05, 1.05])
        ax.set_title('Calibration')

    estimators = [
        (est_name, estimator),
        (est_name + ' isotonic',
         CalibratedClassifierCV(estimator, cv=cv, method='isotonic')),
        (est_name + ' sigmoid',
         CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')),
    ]

    scores = []
    for name, est in estimators:
        est.fit(X[train_idx], y[train_idx])
        proba = est.predict_proba(X[valid_idx])[:, 1]
        scores.append(brier_score_loss(y[valid_idx], proba, pos_label=y.max()))

        # Plot calibration curve
        if plot:
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y[valid_idx], proba, n_bins=15)
            ax.plot(mean_predicted_value, fraction_of_positives,
                    "s-", label=f"{name} ({scores[-1]:1.3f})")

    if plot:
        ax.legend(loc='lower right')
        ax.grid(True)

    best_est = estimators[np.argmax(scores)]
    return best_est[1], np.max(scores)


def estimate_propensity(X: np.ndarray, t: np.ndarray, plot=False):

    scorer = make_scorer(brier_score_loss, greater_is_better=False,
                         needs_proba=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    estimator = LogisticRegressionCV(solver='liblinear', penalty='l1', Cs=10,
                                     cv=cv, scoring=scorer)

    calibrated_estimator, _ = calibrate_classifier(estimator, X, t, plot=plot)

    propensity = calibrated_estimator.predict_proba(X)[:, 1]
    return propensity
