"""
MissX Imputer for Missing Data

Modified code of missforest (https://github.com/stekhoven/missForest)
    - The imputer was modified so that...
        - Custom predictors can be used
        - Predictions are done in parallel (for faster calculation)
        - Delete codes for classification
"""

import warnings
import numpy as np
import copy
from scipy.stats import mode
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed


def _get_mask(X, value_to_mask):
    """Compute the boolean mask X == missing_values."""
    if value_to_mask == "NaN" or np.isnan(value_to_mask):
        return np.isnan(X)
    else:
        return X == value_to_mask


default_rfr = RandomForestRegressor(n_estimators=100, criterion=('mse'),
                                    max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                    min_weight_fraction_leaf=0.0, max_features='auto',
                                    max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    bootstrap=True, oob_score=False, n_jobs=4, random_state=None,
                                    verbose=0, warm_start=False,
                                    )


class MissParallelX(BaseEstimator, TransformerMixin):

    def __init__(self,
                 parallel=-1,
                 fill_initial_by_mean=True,
                 min_iter_count=3,
                 max_iter=10,
                 custom_regressor=None,
                 maintain_original_array=False,
                 decreasing=False,
                 missing_values=np.nan,
                 copy=True,
                 verbose=False,
                 ):
        """
        parallel: nunmber of workers for prediction. if -1, use MAX.
        fill_initial_by_mean: fill initial value with mean
        min_iter_count: minumum iteration of imputation
        max_iter: max iteration of imputation
        custom_regressor: regressor class, which has fit and predict functions (e.g., sklearn)
        maintain_original_array: maintain initial array for prediction
        decreasing=False,
        missing_values=np.nan,
        copy=True,
        verbose=False,
        """
        self.min_iter_count = min_iter_count
        self.fill_initial_by_mean = fill_initial_by_mean
        self.custom_regressor = custom_regressor
        self.maintain_original_array = maintain_original_array

        self.max_iter = max_iter
        self.decreasing = decreasing
        self.missing_values = missing_values
        self.copy = copy
        self.verbose = verbose
        self.parallel = parallel

    def _rfr(self):
        regressor = default_rfr
        return regressor

    def _impute(self, Ximp, mask):
        """The missForest algorithm"""

        # Count missing per column
        col_missing_count = mask.sum(axis=0)

        # Get col and row indices for missing
        missing_rows, missing_cols = np.where(mask)

        if self.num_vars_ is not None:
            # Only keep indices for numerical vars
            keep_idx_num = np.in1d(missing_cols, self.num_vars_)
            missing_num_rows = missing_rows[keep_idx_num]
            missing_num_cols = missing_cols[keep_idx_num]

            # mainatain original X
            self.original_X = copy.copy(Ximp)

            # Make initial guess for missing values
            col_means = np.full(Ximp.shape[1], fill_value=np.nan)
            col_means[self.num_vars_] = self.statistics_.get('col_means')

            if self.fill_initial_by_mean:
                Ximp[missing_num_rows, missing_num_cols] = np.take(
                    col_means, missing_num_cols)

        # 2. misscount_idx: sorted indices of cols in X based on missing count
        misscount_idx = np.argsort(col_missing_count)
        # Reverse order if decreasing is set to True
        if self.decreasing is True:
            misscount_idx = misscount_idx[::-1]

        # 3. While new_gammas < old_gammas & self.iter_count_ < max_iter loop:
        self.iter_count_ = 0
        gamma_new = 0
        gamma_old = np.inf

        self.col_index = np.arange(Ximp.shape[1])

        while self.iter_count_ <= self.min_iter_count or (gamma_new < gamma_old and self.iter_count_ < self.max_iter):
            # Instantiate regression model

            # 4. store previously imputed matrix
            Ximp_old = np.copy(Ximp)
            if self.iter_count_ != 0:
                gamma_old = gamma_new

            self.Ximp = Ximp
            self.mask = mask

            # 5. loop & update
            if self.parallel >= -1:
                Ximp = np.array(Parallel(n_jobs=self.parallel)(
                    [delayed(self._predict)(n) for n in range(self.num_vars_.shape[0])])).T
            else:
                for s in misscount_idx:
                    Ximp = self._predict(s)
                    self.Ximp = Ximp

            if self.num_vars_ is not None:
                gamma_new = np.sum((Ximp[:, self.num_vars_] - Ximp_old[:, self.num_vars_])
                                   ** 2) / np.sum((Ximp[:, self.num_vars_]) ** 2)
            if self.verbose > 0:
                print("Iteration:", self.iter_count_)
            self.iter_count_ += 1

        return Ximp_old

    def _predict(self, s):

        mask = self.mask
        Ximp = copy.copy(self.Ximp)

        if self.custom_regressor is None:
            regressor = self._rfr()
        else:
            regressor = self.custom_regressor

        # Column indices other than the one being imputed
        s_prime = np.delete(self.col_index, s)

        # Get indices of rows where 's' is observed and missing
        obs_rows = np.where(~mask[:, s])[0]
        mis_rows = np.where(mask[:, s])[0]

        # If no missing, then skip
        if len(mis_rows) == 0:
            if self.parallel:
                return Ximp[:, s]
            else:
                return Ximp

        # Get observed values of 's'
        yobs = Ximp[obs_rows, s]

        # add original array
        if self.maintain_original_array:
            Ximp2 = np.concatenate(
                [Ximp, np.delete(self.original_X, s, axis=1)], axis=1)
        else:
            Ximp2 = Ximp

        # Get 'X' for both observed and missing 's' column
        xobs = Ximp2[np.ix_(obs_rows, s_prime)]
        xmis = Ximp2[np.ix_(mis_rows, s_prime)]

        regressor.fit(X=xobs, y=yobs)

        # predict ymis(s) using xmis(x)
        ymis = regressor.predict(xmis)

        Ximp[mis_rows, s] = ymis.reshape(-1)

        if self.parallel:
            return Ximp[:, s]

        return Ximp

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : object
            Returns self.
        """

        # Check data integrity and calling arguments
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True

        X = check_array(X, accept_sparse=False, dtype=np.float64,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            #raise ValueError("One or more columns have all rows missing.")
            # fill with zeros
            miss_rows = np.where(np.all(mask, axis=0) == True)[0]
            for i in miss_rows:
                X[:, i] = 0

        # Identify numerical variables
        num_vars = np.arange(X.shape[1])
        num_vars = num_vars if len(num_vars) > 0 else None

        # First replace missing values with NaN if it is something else
        if self.missing_values not in ['NaN', np.nan]:
            X[np.where(X == self.missing_values)] = np.nan

        # Now, make initial guess for missing values
        col_means = np.nanmean(
            X[:, num_vars], axis=0) if num_vars is not None else None

        self.num_vars_ = num_vars
        self.statistics_ = {"col_means": col_means}

        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            The input data to complete.

        Returns
        -------
        X : {array-like}, shape = [n_samples, n_features]
            The imputed dataset.
        """
        # Confirm whether fit() has been called
        check_is_fitted(self, ["num_vars_", "statistics_"])

        # Check data integrity
        force_all_finite = False if self.missing_values in ["NaN",
                                                            np.nan] else True
        X = check_array(X, accept_sparse=False, dtype=np.float32,
                        force_all_finite=force_all_finite, copy=self.copy)

        # Check for +/- inf
        if np.any(np.isinf(X)):
            raise ValueError("+/- inf values are not supported.")

        # Check if any column has all missing
        mask = _get_mask(X, self.missing_values)
        if np.any(mask.sum(axis=0) >= (X.shape[0])):
            #raise ValueError("One or more columns have all rows missing.")
            # fill with zeros
            miss_rows = np.where(np.all(mask, axis=0) == True)[0]
            for i in miss_rows:
                X[:, i] = 0

        # Get fitted X col count and ensure correct dimension
        n_cols_fit_X = (0 if self.num_vars_ is None else len(self.num_vars_))
        _, n_cols_X = X.shape

        if n_cols_X != n_cols_fit_X:
            raise ValueError("Incompatible dimension between the fitted "
                             "dataset and the one to be transformed.")

        # Check if anything is actually missing and if not return original X
        mask = _get_mask(X, self.missing_values)
        if not mask.sum() > 0:
            warnings.warn("No missing value located; returning original "
                          "dataset.")
            return X

        # Call missForest function to impute missing
        X = self._impute(X, mask)

        # Return imputed dataset
        return X

    def fit_transform(self, X, y=None, **fit_params):
        """Fit MissForest and impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        X : {array-like}, shape (n_samples, n_features)
            Returns imputed dataset.
        """
        return self.fit(X, **fit_params).transform(X)
