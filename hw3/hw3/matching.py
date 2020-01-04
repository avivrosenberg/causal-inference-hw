import math

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted


class MatchingEstimator(BaseEstimator):
    """
    An sklearn-style estimator that performs matching between two
    groups of covariates, e.g. treated and control. Based on a KNN approach.
    """
    METHODS = {'mahalanobis', 'euclidean', 'cosine', 'random'}

    def __init__(self, method='cosine'):
        self.method = method

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
            self.knn_ = NearestNeighbors(n_neighbors=1, **metric)
            self.knn_.fit(X)

        self.fitted_X_ = X
        return self

    def transform(self, X):
        """
        Finds and returns the closest sample from the fitted data to the
        given query data.
        @param X: The query data, shape (N, d).
        @return: (Xref_matched, idx) Closest samples from the reference
        (fitted data) of shape (N, d) and the indices in the reference data
        they correspond to.
        """
        check_is_fitted(self, ['fitted_X_'])

        _, idx = self.kneighbors(X, n_neighbors=1)
        idx = idx[:, 0]

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

    def kneighbors(self, X, n_neighbors=1):

        if self.method == 'random':
            idx = np.arange(self.fitted_X_.shape[0])
            idx_matches = np.random.choice(idx, size=X.shape[0], replace=True)
            dists = euclidean_distances(X, self.fitted_X_[idx_matches])
            return dists, idx_matches.reshape(-1, 1)

        return self.knn_.kneighbors(X, n_neighbors=n_neighbors,
                                    return_distance=True)

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

        # For random method, all ref samples will be matched
        if self.method == 'random':
            Xref_m, Xref_m_idx = self.transform(Xquery)
            Xquery_m = Xquery
            Xquery_m_idx = np.arange(Xquery.shape[0])
            dists = euclidean_distances(Xref_m, Xquery_m)
            return Xref_m, Xquery_m, Xref_m_idx, Xquery_m_idx, dists

        dist, idx = self.kneighbors(Xquery)  # (Nquery, 1) indices into Xref
        dist = dist[:, 0]
        idx = idx[:, 0]

        max_dist = np.mean(dist) * tol
        valid_dist_idx = dist < max_dist

        Xref_m_idx = idx[valid_dist_idx]  # (M, 1) queries with a valid match
        Xref_m = Xref[Xref_m_idx]

        Xquery_m_idx = np.arange(Xquery.shape[0])[valid_dist_idx]
        Xquery_m = Xquery[Xquery_m_idx]

        return Xref_m, Xquery_m, Xref_m_idx, Xquery_m_idx, dist[valid_dist_idx]
