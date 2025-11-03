# encoder.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SimpleTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10):
        self.smoothing = smoothing

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y, index=X.index)
        self.global_mean_ = y.mean()
        self.mappings_ = {}
        self.feature_names_in_ = list(X.columns)

        for i, col in enumerate(X.columns):
            df_temp = pd.DataFrame({'cat': X[col], 'target': y})
            stats = df_temp.groupby('cat')['target'].agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + self.global_mean_ * self.smoothing) / (stats['count'] + self.smoothing)
            self.mappings_[i] = smooth.to_dict()
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        out = np.empty_like(X, dtype=float)
        for i, col in enumerate(X.columns):
            mapping = self.mappings_.get(i, {})
            default = self.global_mean_
            out[:, i] = X[col].map(mapping).fillna(default).values
        return out

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_