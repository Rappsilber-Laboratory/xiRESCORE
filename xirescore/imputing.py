from typing import Union

import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator


class InfImputer:
    def __init__(self,
                 base_imputer: Union[SimpleImputer, KNNImputer, MissingIndicator],
                 ranges: list[tuple[float, float]],
                 inf_factor=0.1):
        self.base_imputer = base_imputer
        self.ranges = ranges
        self.abs_ranges = np.array([
            max_val - min_val
            for min_val, max_val in ranges
        ])
        self.pos_inf_impute_values = np.array([
            max_val+abs_range*inf_factor
            for (_, max_val), abs_range in zip(ranges, self.abs_ranges)
        ])
        self.neg_inf_impute_values = np.array([
            max_val+abs_range*inf_factor
            for (_, max_val), abs_range in zip(ranges, self.abs_ranges)
        ])

    def fit(self, X):
        X = np.array(X)
        min_inpute_arr = np.array([
            self.neg_inf_impute_values
            for _ in range(X.shape[0])
        ])
        max_inpute_arr = np.array([
            self.pos_inf_impute_values
            for _ in range(X.shape[0])
        ])
        X = np.where(X==-np.inf, min_inpute_arr, X)
        X = np.where(X==np.inf,  max_inpute_arr, X)
        # Fill in empty columns
        empty_feats = np.isnan(np.max(X, axis=0))
        X[:,empty_feats] = 0
        self.base_imputer.fit(X)

    def transform(self, X):
        X = np.array(X)
        min_inpute_arr = np.array([
            self.neg_inf_impute_values
            for _ in range(X.shape[0])
        ])
        max_inpute_arr = np.array([
            self.pos_inf_impute_values
            for _ in range(X.shape[0])
        ])
        X = np.where(X==-np.inf, min_inpute_arr, X)
        X = np.where(X==np.inf,  max_inpute_arr, X)
        # Fill in empty columns
        empty_feats = np.isnan(np.max(X, axis=0))
        X[:,empty_feats] = 0
        return self.base_imputer.transform(X)

    def inverse_transform(self, X):
        min_inpute_arr = np.array([
            [x for x, _ in self.ranges]
            for _ in X
        ])
        max_inpute_arr = np.array([
            [x for _, x in self.ranges]
            for _ in X
        ])
        X[:] = np.where(X==min_inpute_arr, np.full(X.shape, -np.inf), X)
        X[:] = np.where(X==max_inpute_arr, np.full(X.shape, np.inf),  X)
        return self.base_imputer.inverse_transform(X)

class InfScaler:
    def __init__(self,
                 base_scaler: Union[SimpleImputer, KNNImputer, MissingIndicator],
                 ranges: list[tuple[float, float]],
                 inf_factor=0.1):
        self.base_scaler = base_scaler
        self.ranges = ranges
        self.abs_ranges = np.array([
            max_val - min_val
            for min_val, max_val in ranges
        ])
        self.pos_inf_impute_values = np.array([
            max_val + abs_range * inf_factor
            for (_, max_val), abs_range in zip(ranges, self.abs_ranges)
        ])
        self.neg_inf_impute_values = np.array([
            max_val + abs_range * inf_factor
            for (_, max_val), abs_range in zip(ranges, self.abs_ranges)
        ])

    def fit(self, X):
        min_inpute_arr = np.array([
            self.neg_inf_impute_values
            for _ in range(X.shape[0])
        ])
        max_inpute_arr = np.array([
            self.pos_inf_impute_values
            for _ in range(X.shape[0])
        ])
        X = np.where(X == -np.inf, min_inpute_arr, X)
        X = np.where(X == np.inf, max_inpute_arr, X)
        self.base_scaler.fit(X)

    def transform(self, X):
        min_inpute_arr = np.array([
            self.neg_inf_impute_values
            for _ in range(X.shape[0])
        ])
        max_inpute_arr = np.array([
            self.pos_inf_impute_values
            for _ in range(X.shape[0])
        ])
        X = np.where(X == -np.inf, min_inpute_arr, X)
        X = np.where(X == np.inf, max_inpute_arr, X)
        return self.base_scaler.transform(X)

    def inverse_transform(self, X):
        min_inpute_arr = np.array([
            [x for x, _ in self.ranges]
            for _ in X
        ])
        max_inpute_arr = np.array([
            [x for _, x in self.ranges]
            for _ in X
        ])
        X[:] = np.where(X == min_inpute_arr, np.full(X.shape, -np.inf), X)
        X[:] = np.where(X == max_inpute_arr, np.full(X.shape, np.inf), X)
        return self.base_scaler.inverse_transform(X)
