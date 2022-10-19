
import numpy as np
from scipy.sparse import issparse, csc_matrix
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted


def _sparse_corr(X):
    n = X.shape[0]
    C = ((X.T * X - (sum(X).T * sum(X) / n)) / (n - 1)).todense()
    V = np.sqrt(np.mat(np.diag(C)).T * np.mat(np.diag(C)))
    cor = np.divide(C, V + 1e-119)
    return np.array(cor)


class CorrelationFilter(BaseEstimator, SelectorMixin):
    """Manual column feature selector

    Parameters
    ----------
    cols : array-like (default=None)
        A list specifying the feature indices to be selected.
    """
    def __init__(self, corr_threshold: float = 0.9):
        self.n_features_ = None
        self.mask = None
        self.corr_threshold = corr_threshold

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        Returns
        ---------
        self : object
            Returns self.
        """
        self.mask = self._find_cols_to_keep(X=X)
        self.n_features_ = X.shape[1]
        return self

    def _find_cols_to_keep(self, X):
        corr = _sparse_corr(X) if issparse(X) else np.corrcoef(X, rowvar=False)
        upper = np.abs(corr * np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_keep = np.array(
            [column for column in range(upper.shape[1]) if not any(upper[:, column] >= self.corr_threshold)])
        return to_keep

    def _get_support_mask(self):
        check_is_fitted(self, ('n_features_', 'mask'))
        return np.array([col in self.mask for col in range(0, self.n_features_)])


if __name__ == '__main__':
    # Test behavior of this class

    # Generate sample dataset with correlated and uncorrelated features
    x1 = np.zeros((1000, 1))
    idx = np.random.randint(0, 1000, size=10)
    x1[idx] = 1

    x2 = np.zeros((1000, 1))
    idx = np.random.randint(0, 1000, size=10)
    x2[idx] = -1

    arr = np.concatenate([x1, x1, -x1, x2, x2, -x2], axis=1)

    corr_filter = CorrelationFilter(corr_threshold=0.95)

    # Sparse version
    corr_filter.fit(csc_matrix(arr))
    assert corr_filter.mask.tolist() == [0, 3]
    filtred_arr_1 = corr_filter.transform(csc_matrix(arr))

    # Dense version
    corr_filter.fit(arr)
    filtred_arr_2 = corr_filter.transform(arr)

    assert (filtred_arr_1 == filtred_arr_2).all()
