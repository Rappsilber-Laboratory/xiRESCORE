from logging import Logger

from sklearn.base import ClassifierMixin
from sklearn import preprocessing, impute
import polars as pl

from xirescore.feature_extracting import get_features


def get_transformers(df: pl.DataFrame, options: dict):
    """
    Normalize the features and drop NaN-values if necessary.
    """
    filter_nan = options['rescoring']['imputer'] is None
    features = get_features(df, options, filter_nan)
    df_features = df.select(features)

    Scaler: ClassifierMixin.__class__ = getattr(preprocessing, options['rescoring']['scaler'])
    scaler_options = options['rescoring']['scaler_params']

    imputer = None
    if options['rescoring']['imputer'] is not None:
        # Impute missing feature values
        imputer_class = getattr(impute, options['rescoring']['imputer'])
        imputer = imputer_class(
            **options['rescoring']['imputer_kwargs']
        )
        imputer.fit(
            df_features
        )

    scaler = Scaler(**scaler_options)
    scaler.fit(df_features)

    return imputer, scaler, features
