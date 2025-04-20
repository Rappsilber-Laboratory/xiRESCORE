from logging import Logger

from sklearn.base import ClassifierMixin
from sklearn import preprocessing
import polars as pl

from xirescore.feature_extracting import get_features


def get_scaler(df: pl.DataFrame, options: dict):
    """
    Normalize the features and drop NaN-values if necessary.
    """
    features = get_features(df, options)
    df_features = df.select(features)

    Scaler: ClassifierMixin.__class__ = getattr(preprocessing, options['rescoring']['scaler'])
    scaler_options = options['rescoring']['scaler_params']
    scaler = Scaler(**scaler_options)
    scaler.fit(df_features)

    return scaler, features
