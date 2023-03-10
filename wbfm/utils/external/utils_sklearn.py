import numpy as np
from sklearn.model_selection._split import _BaseKFold, KFold, _validate_shuffle_split, TimeSeriesSplit
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples


class RollingOriginForwardValidation(_BaseKFold):
    """Slight modification of TimeSeriesSplit to include all test data to the end of the dataset"""
    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )

        # Make sure we have enough samples for the given split parameters
        if n_folds > n_samples:
            raise ValueError(
                f"Cannot have number of folds={n_folds} greater"
                f" than the number of samples={n_samples}."
            )
        if n_samples - gap - (test_size * n_splits) <= 0:
            raise ValueError(
                f"Too many splits={n_splits} for number of samples"
                f"={n_samples} with test_size={test_size} and gap={gap}."
            )

        indices = np.arange(n_samples)
        test_starts = range(n_samples - n_splits * test_size, n_samples, test_size)

        for test_start in test_starts:
            train_end = test_start - gap
            if self.max_train_size and self.max_train_size < train_end:
                yield (
                    indices[train_end - self.max_train_size:train_end],
                    indices[test_start:],
                )
            else:
                yield (
                    indices[:train_end],
                    indices[test_start:],
                )


class LastBlockForwardValidation(TimeSeriesSplit):
    """
    Exposes just the indexing of test_train_split, with shuffle=False

    Returns these indices twice in order to match the API of KFold
    """

    def split(self, X, y=None, groups=None):
        n_samples = _num_samples(X)
        test_size = 1 / self.n_splits
        train_size = 1 - test_size
        n_train, n_test = _validate_shuffle_split(n_samples, test_size, train_size)
        # err
        train = np.arange(n_train)
        test = np.arange(n_train, n_train + n_test)

        # return [(train, test)]

        for i in range(2):
            yield (
                train,
                test
            )


def middle_40_cv_split(trace_len):
    """
    Do basic split: test is middle 40%, rest is train

    Note that this is a function because there is only a single split

    Parameters
    ----------
    trace_len

    Returns
    -------

    """
    len_third = int(trace_len / 3)
    ind_test = list(range(trace_len))
    ind_train = ind_test[:len_third]
    ind_train.extend(ind_test[-len_third:])
    [ind_test.remove(i) for i in ind_train]
    len(ind_test), len(ind_train)
    return ind_test, ind_train
