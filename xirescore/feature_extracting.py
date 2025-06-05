import logging
from logging import Logger

import polars as pl


logger = logging.getLogger(__name__)

def get_features(df: pl.DataFrame, options: dict, filter_nan=True):
    features_const = options['input']['columns']['features']
    feat_prefix = options['input']['columns']['feature_prefix']
    features_prefixes = [
        c for c in df.columns if str(c).startswith(feat_prefix)
    ]
    features = list(set(features_const + features_prefixes))

    absent_features = [
        f
        for f in features
        if f not in df.columns
    ]
    nan_features = []
    if filter_nan:
        nan_features = [
            f
            for f in features
            if (f not in absent_features) and any(df[f].is_null())
        ]
    features = [
        f
        for f in features
        if f not in (absent_features + nan_features)
    ]

    if len(nan_features) > 0:
        logger.warning(f"Dropped features with NaN values: {nan_features}")
    if len(absent_features) > 0:
        logger.warning(f"Did not find some features: {absent_features}")

    return features
