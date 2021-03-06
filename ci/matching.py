import math

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import paired_euclidean_distances as d_l2
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted


def ssmd_dist(X1: np.ndarray, X2: np.ndarray):
    """
    SSMD distance metric between pairs of random variables.
    https://en.wikipedia.org/wiki/Strictly_standardized_mean_difference
    We assume each dataset has N samples from d random variables and we want
    to compare variable j=[0,d-1] between the datasets.
    To make this a distance, we return the absolute value of the SSMD for
    each variable.

    @param X1: Data from population 1 (e.g. treat), of shape (N1, d).
    @param X2: Data from population 2 (e.g. control), of shape (N2, d).
    @return: The SSMD distance (abs value) for each variable. Shape (d,).
    """
    assert X1.shape[1] == X2.shape[1]

    # If number of samples is not equal, trim the longer one.
    imax = min(X1.shape[0], X2.shape[0])
    X1, X2 = X1[:imax], X2[:imax]

    dX = X1 - X2
    mu = np.mean(dX, axis=0, keepdims=False)
    sigma2 = np.var(dX, axis=0, keepdims=False)

    # Prevent division by zero
    nonzero_var = sigma2 > 0
    return np.abs(mu[nonzero_var] / np.sqrt(sigma2[nonzero_var]))


def wass_dist(X1: np.ndarray, X2: np.ndarray):
    """
    Wasserstein distance metric between pairs of random variables.
    https://en.wikipedia.org/wiki/Wasserstein_metric
    We assume each dataset has N samples from d random variables and we want
    to compare variable j=[0,d-1] between the datasets.

    @param X1: Data from population 1 (e.g. treat), of shape (N1, d).
    @param X2: Data from population 2 (e.g. control), of shape (N2, d).
    @return: The Wasserstein distance (abs value) for each variable, (d,).
    """
    assert X1.shape[1] == X2.shape[1]
    d = X1.shape[1]

    return np.array([
        wasserstein_distance(X1[:, j], X2[:, j]) for j in range(d)
    ])


def propensity_matching(X: np.ndarray, t: np.ndarray, p: np.ndarray,
                        match_to=0, k=1, tol=math.inf):
    """
    Matching based on propensity scores.
    @param X: Covariates of shape (N, d).
    @param t: Treatment assignment values of shape (N,). We assume binary
    treatment where t[i]=1 means treatment and t[i]=0 means control.
    @param p: Propensity scores of shape (N,).
    @param match_to: Whether to match to the control group (t=0) or to the
    treatment group (t=1). The group with t==match_to is the "reference"
    @param k: Maximal number of nearest neighbors from the reference to return
    per sample from the query group.
    @param tol: Tolerance for matching as a fraction of the mean distance
    between pairs with the closest propensity scores. E.g. if tol=0.9 then
    only matching pairs for which the propensity distance is less than 90%
    of the average distance among all pairs will be returned.
    @return: Tuple of (X_ref_m, X_query_m, idx_ref_m, idx_query_m, dists)
    where these correspond to
    the matched pairs of samples from the reference and query groups groups
    respectively, their respective indices, and the distances between their
    covariates (calculated according to the relevant metric for `method`).
    """
    # We'll use propensity as 1d-covariates for matching with euclidean
    # distance (just a difference in the 1d case).
    p = p.reshape(-1, 1)
    _, _, idx_ref_m, idx_query_m, dists = covariate_matching(
        X=p, t=t, method='euclidean', match_to=match_to, k=k, tol=tol
    )

    # Get the matching sample's covariates
    X_ref_m = X[idx_ref_m]
    X_query_m = X[idx_query_m]

    return X_ref_m, X_query_m, idx_ref_m, idx_query_m, dists


def covariate_matching(X: np.ndarray, t: np.ndarray, method,
                       match_to=0, k=1, tol=math.inf):
    """
    k-NN Matching based on covariate values.
    @param X: Covariates of shape (N, d).
    @param t: Treatment assignment values of shape (N,). We assume binary
    treatment where t[i]=1 means treatment and t[i]=0 means control.
    @param method: Method to use for calculating the distance metric.
    Supported methods are listed in MatchingEstimator.METHODS.
    @param match_to: Whether to match to the control group (t=0) or to the
    treatment group (t=1). The group with t==match_to is the "reference"
    group and the other group (t==(1-match_to)) is the "query" group.
    @param k: Maximal number of nearest neighbors from the reference to return
    per sample from the query group.
    @param tol: Tolerance for matching as a fraction of the mean distance
    between pairs with the closest propensity scores. E.g. if tol=0.9 then
    only matching pairs for which the propensity distance is less than 90%
    of the average distance among all pairs will be returned.
    @return: Tuple of (X_ref_m, X_query_m, idx_ref_m, idx_query_m, dists)
    where these correspond to
    the matched pairs of samples from the reference and query groups groups
    respectively, their respective indices, and the distances between their
    covariates (calculated according to the relevant metric for `method`).
    """
    # Assume binary treatment of t=0 or t=1
    assert match_to == 0 or match_to == 1

    # Create numeric (non boolean) indices
    # Query = what we match, Ref = what we match it to
    idx = np.arange(X.shape[0])
    idx_query, idx_ref = idx[t == (1 - match_to)], idx[t == match_to]
    X_query, X_ref = X[idx_query], X[idx_ref]

    matcher = MatchingEstimator(method=method, k=k)
    X_ref_m, X_query_m, idx_ref_m, idx_query_m, dists = matcher.match(
        Xref=X_ref, Xquery=X_query, tol=tol
    )

    # Convert indices to be relative the entire dataset, not the groups
    idx_ref_m, idx_query_m = idx_ref[idx_ref_m], idx_query[idx_query_m]

    return X_ref_m, X_query_m, idx_ref_m, idx_query_m, dists


class MatchingEstimator(BaseEstimator):
    """
    An sklearn-style estimator that performs matching between two
    groups of covariates, e.g. treated and control. Based on a KNN approach.
    """
    METHODS = {'mahalanobis', 'euclidean', 'cosine', 'random'}

    def __init__(self, method='cosine', k=1):
        assert method in self.METHODS
        assert k > 0
        self.method = method
        self.k = k

    def fit(self, X, y=None):
        """
        Fits the estimator to the data.
        @param X: The reference samples to match to, shape (N, d)
        @param y: Ignored.
        @return: self.
        """

        metric = dict(metric=self.method)
        if self.method == 'mahalanobis':
            C = np.cov(X, rowvar=False)
            metric['metric_params'] = dict(VI=np.linalg.inv(C))

        if self.method != 'random':
            self.knn_ = NearestNeighbors(**metric)
            self.knn_.fit(X)

        self.fitted_X_ = X
        return self

    def transform(self, X):
        """
        Finds and returns the k-nearest samples from the fitted data to the
        given query data.
        @param X: The query data, shape (N, d).
        @return: (Xref_matched, idx) k-Closest samples from the reference
        (fitted data) of shape (N, k, d) and the indices in the reference data
        they correspond to.
        """
        check_is_fitted(self, ['fitted_X_'])

        _, idx = self.kneighbors(X)

        return self.fitted_X_[idx], idx

    def score(self, X):
        """
        Returns reciprocal of average distance for a group of query samples
        compared to the reference (fitted) samples.
        @param X: Query samples.
        @return: One over the average distance.
        """
        dists, _ = self.kneighbors(X)
        return 1 / np.mean(dists)

    def kneighbors(self, X):
        """
        Returns k closest neighbors to X in the reference (fitted) data and
        their indices in the reference data.
        @param X: Query data of shape (N, d).
        @param k: Number of neightbors to fetch.
        @return: Tuple (dists, idx) where both are shape (N, k).
        """
        k = self.k
        if self.method == 'random':
            idx = np.arange(self.fitted_X_.shape[0])
            idx_m = np.random.choice(idx, size=(X.shape[0], k), replace=True)
            dists = np.hstack([
                d_l2(X, self.fitted_X_[idx_m[:, j]]).reshape(-1, 1)
                for j in range(k)
            ])

            return dists, idx_m

        return self.knn_.kneighbors(X, k, return_distance=True)

    def match(self, Xref, Xquery, tol=math.inf):
        """
        Returns reference samples that match the query samples within a
        given tolerance value.
        @param Xref: Reference samples. These will be used to match to.
        Shape should be (Nref, d).
        @param Xquery: Query samples. These will be matched to the reference
        samples. Shape should be (Nquery, d).
        @param tol: Tolerance for matching, relative to mean distance. Maximal
        allowed distance between two samples to be considered a match will
        be mean * tol.
        @return: Tuple of (Xref_m, X_query_m, Xref_m_idx, Xquery_m_idx, dist).
        Xref_m/Xquery_m are both of shape (M, d) and represents pairs of
        samples that match within the required tolerance. Xref_m_idx and
        Xquery_m_idx are the indices of the matched samples within each group.
        Finally, dist contains the distance between each matched pair.
        """
        assert tol > 0
        self.fit(Xref)

        dist, idx = self.kneighbors(Xquery)  # (Nquery, k)

        if self.method == 'random':
            valid_dist_idx = np.ones_like(dist, dtype=np.bool)
        else:
            max_dist = np.mean(dist) * tol
            valid_dist_idx = dist < max_dist

        Xref_m_idx = idx[valid_dist_idx]  # (M, 1) queries with a valid match
        Xref_m = Xref[Xref_m_idx]

        Xquery_m_idx = np.arange(Xquery.shape[0]).reshape(-1, 1)  # (N,1)
        Xquery_m_idx = np.repeat(Xquery_m_idx, self.k, axis=1)  # (N,k)
        Xquery_m_idx = Xquery_m_idx[valid_dist_idx]
        Xquery_m = Xquery[Xquery_m_idx]

        return Xref_m, Xquery_m, Xref_m_idx, Xquery_m_idx, dist[valid_dist_idx]
