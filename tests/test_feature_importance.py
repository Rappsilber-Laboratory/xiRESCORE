#!/usr/bin/env python

"""Tests for feature importance"""
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

@pytest.mark.df
def test_feature_importance_lr():
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
    rescorer = XiRescore(
        input_path=df,
        options=options,
    )
    rescorer.run()
    assert rescorer.get_rescored_output() is not None
    assert rescorer.pca is not None
    importances = rescorer.get_feature_importance()
    assert importances.shape[0] == len(rescorer.models)
    assert importances.shape[0] == len(rescorer.train_features)


@pytest.mark.df
def test_feature_importance_lr():
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
    rescorer = XiRescore(
        input_path=df,
        options=options,
    )
    rescorer.run()
    assert rescorer.get_rescored_output() is not None
    assert rescorer.pca is not None
    importances = rescorer.get_feature_importance()
    assert importances.shape[0] == len(rescorer.models)
    assert importances.shape[1] == len(rescorer.train_features)


@pytest.mark.df
def test_feature_importance_svc():
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
            'spectra_batch_size': 25_000,  # Rescore in 4 batches
            'train_selection_mode': 'self-targets-capped-decoys',
            'scaler': 'QuantileTransformer',
            'scaler_params': {
                'output_distribution': 'normal'
            },
            'model_class': 'svm',
            'model_name': 'SVC',
            'model_params': {
                'kernel': ['rbf'],
                'probability': [True],
            },
            'random_seed': 123456
        }
    }

    random.seed(0)
    np.random.seed(0)
    pl.set_random_seed(0)
    rescorer = XiRescore(
        input_path=df,
        options=options,
    )
    rescorer.run()
    assert rescorer.get_rescored_output() is not None
    assert rescorer.pca is None
    importances = rescorer.get_feature_importance()
    assert importances.shape[0] == len(rescorer.models)
    assert importances.shape[1] == len(rescorer.train_features)

@pytest.mark.df
def test_feature_importance_tree():
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
            'spectra_batch_size': 25_000,  # Rescore in 4 batches
            'train_selection_mode': 'self-targets-capped-decoys',
            'scaler': 'QuantileTransformer',
            'scaler_params': {
                'output_distribution': 'normal'
            },
            'model_class': 'tree',
            'model_name': 'DecisionTreeClassifier',
            'model_params': {},
            'random_seed': 123456
        }
    }

    random.seed(0)
    np.random.seed(0)
    pl.set_random_seed(0)
    rescorer = XiRescore(
        input_path=df,
        options=options,
    )
    rescorer.run()
    assert rescorer.get_rescored_output() is not None
    assert rescorer.pca is None
    importances = rescorer.get_feature_importance()
    assert importances.shape[0] == len(rescorer.models)
    assert importances.shape[1] == len(rescorer.train_features)
