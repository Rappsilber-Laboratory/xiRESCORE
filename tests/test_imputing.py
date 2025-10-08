#!/usr/bin/env python

"""Tests for `xirescore` package."""
import random
import sys
import pandas as pd
import polars as pl
import pytest
import logging
import tempfile
import subprocess
import os
import numpy as np
import yaml
from xirescore.XiRescore import XiRescore
from xirescore.readers import read_value_ranges


@pytest.mark.df
def test_imputing():
    random.seed(0)
    np.random.seed(0)
    df = pl.read_parquet('./tests/fixtures/test_data.parquet')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.info('Start full DF rescoring test')

    options = {
        'input': {
            'csm_id': ['__index_level_0__'],
            'columns': {
                'features': [
                    'match_score',
                    'better_score',
                    'worse_score',
                    'useless_score_uni',
                    'useless_score_norm',
                    'conditional_score',
                ],
                'score': 'match_score',
            },
        },
        'rescoring': {
            'imputer': 'SimpleImputer',
            'imputer_kwargs': {'strategy': 'mean'},
            'pca_n_components': 'mle',
            'spectra_batch_size': 25_000,  # Rescore in 4 batches
            'train_selection_mode': 'self-targets-capped-decoys',
            'scaler': 'QuantileTransformer',
            'scaler_params': {
                'output_distribution': 'normal'
            },
            'model_params': {
                "C": [10],
                "solver": ["liblinear"],
                "penalty": ["l2"],
                "class_weight": ["balanced"],
                "random_state": [0],
            },
            'random_seed': 123456
        }
    }

    random.seed(0)
    np.random.seed(0)
    pl.set_random_seed(0)
    df = df.with_columns(
        delete_value_mask = np.random.choice([True, False], size=df.height)
    ).with_columns(
        better_score=pl.when(
            pl.col('delete_value_mask')
        ).then(
            pl.lit(None)
        ).otherwise(
            pl.col('better_score')
        )
    )
    rescorer = XiRescore(
        input_path=df,
        options=options,
    )
    rescorer.run()
    assert rescorer.imputer is not None
    assert 'better_score' in rescorer.train_features
    assert rescorer.get_rescored_output() is not None

@pytest.mark.df
def test_inf_ranges():
    ranges = read_value_ranges(
        './tests/fixtures/test_data.parquet',
        columns=[
            'match_score',
            'better_score',
            'worse_score',
            'useless_score_uni',
            'useless_score_norm',
            'conditional_score',
        ]
    )
    for feat, (min_val, max_val) in ranges.items():
        assert isinstance(feat, str)
        assert isinstance(np.float64(min_val), np.float64)
        assert isinstance(np.float64(max_val), np.float64)

@pytest.mark.df
def test_imputing_inf_df():
    df = pl.read_parquet('./tests/fixtures/test_data.parquet')
    features = [
            'match_score',
            'better_score',
            'worse_score',
            'useless_score_uni',
            'useless_score_norm',
            'conditional_score',
        ]
    df = df.with_columns(
        useless_score_uni=pl.when(pl.arange(pl.len())==0).then(
            pl.lit(np.inf)
        ).otherwise(
            pl.col('useless_score_uni')
        )
    )
    ranges = read_value_ranges(
        df
    )
    assert all([
        f in ranges.keys()
        for f in features
    ])
    # Inf values should be ignored
    assert ranges['useless_score_uni'][1] != np.inf
    for feat, (min_val, max_val) in ranges.items():
        assert isinstance(feat, str)
        assert isinstance(np.float64(min_val), np.float64)
        assert isinstance(np.float64(max_val), np.float64)
