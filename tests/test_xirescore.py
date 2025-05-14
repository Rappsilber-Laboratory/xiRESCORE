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


@pytest.mark.parquet
def test_full_parquet_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full parquet rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ]
                }
            },
            'rescoring': {
                'spectra_batch_size': 25_000
            }
        }

        rescorer = XiRescore(
            input_path='./tests/fixtures/test_data.parquet',
            output_path=f'{tmpdirname}/result.parquet',
            options=options,
        )
        rescorer.run()
        df_in = pd.read_parquet('./tests/fixtures/test_data.parquet')
        df_out = pd.read_parquet(f'{tmpdirname}/result.parquet')
        assert len(df_in) == len(df_out)


@pytest.mark.parquet
def test_linear_filter():
    random.seed(0)
    np.random.seed(0)
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full parquet rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ]
                }
            },
            'rescoring': {
                'spectra_batch_size': 25_000
            }
        }

        df = pd.read_parquet('./tests/fixtures/test_data.parquet')

        df_linear = df.sample(100)
        df_linear['base_sequence_p2'] = None
        df = pd.concat([
            df,
            df_linear
        ])

        df_linear = df.sample(100)
        df_linear['base_sequence_p2'] = ''
        df = pd.concat([
            df,
            df_linear
        ])

        df.to_parquet(f'{tmpdirname}/input.parquet')

        rescorer = XiRescore(
            input_path=f'{tmpdirname}/input.parquet',
            output_path=f'{tmpdirname}/result.parquet',
            options=options,
        )
        rescorer.run()
        df_out = pd.read_parquet(f'{tmpdirname}/result.parquet')
        assert len(df) == len(df_out)+200


@pytest.mark.parquet
@pytest.mark.svc
@pytest.mark.slow
def test_full_svc_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full parquet rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ]
                }
            },
            'rescoring': {
                'model_class': 'svm',
                'model_name': 'SVC',
                'model_params': {
                    'kernel': ['rbf'],
                    'gamma': [1e-2, 1e-3, 1e-4, "auto"],
                    'C': [5, 8, 10, 15],
                    'probability': [True],
                    "class_weight": ["balanced", None, {0: 2, 1: 1}],
                    "tol": [100*np.finfo(np.double).eps]
                }
            }
        }

        rescorer = XiRescore(
            input_path='./tests/fixtures/test_data.parquet',
            output_path=f'{tmpdirname}/result.parquet',
            options=options,
        )
        rescorer.run()
        df_in = pd.read_parquet('./tests/fixtures/test_data.parquet')
        df_out = pd.read_parquet(f'{tmpdirname}/result.parquet')
        assert len(df_in) == len(df_out)


@pytest.mark.csv
def test_full_csv_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full CSV rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ]
                }
            },
            'rescoring': {
                'spectra_batch_size': 25_000  # Rescore in 4 batches
            }
        }

        rescorer = XiRescore(
            input_path='./tests/fixtures/test_data.csv.gz',
            output_path=f'{tmpdirname}/result.csv.gz',
            options=options,
        )
        rescorer.run()
        df_in = pd.read_csv('./tests/fixtures/test_data.csv.gz')
        df_out = pd.read_csv(f'{tmpdirname}/result.csv.gz')
        assert len(df_in) == len(df_out)


@pytest.mark.df
def test_full_df_rescoring():
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
            'model_params': {
                "C": np.logspace(-3, 2, 6),
                "solver": ["liblinear"],
                "penalty": ["l1", "l2"],
                "class_weight": ["balanced", None, {0: 2, 1: 1}],
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
    df_out = rescorer.get_rescored_output()
    assert len(df) == len(df_out)

    # See if runs are reproducible
    random.seed(0)
    np.random.seed(0)
    pl.set_random_seed(0)
    rescorer = XiRescore(
        input_path=df,
        options=options,
    )
    rescorer.run()
    df_out2 = rescorer.get_rescored_output()
    df_comp1 = df_out.with_columns(
        (~pl.selectors.numeric()).hash(),
    )
    df_comp2 = df_out2.with_columns(
        (~pl.selectors.numeric()).hash(),
    )
    assert all(np.isclose(
        df_comp1.to_numpy().flatten(),
        df_comp2.to_numpy().flatten(),
        equal_nan=True
    ))

    rescorer2 = XiRescore(
        input_path=None,
        options=options,
    )
    rescorer2.train_features = rescorer.train_features
    rescorer2.scaler = rescorer.scaler
    rescorer2.models = rescorer.models
    rescorer2.train_df = pl.DataFrame(schema=df.schema)
    df_out3 = rescorer2.rescore_df(
        df_out2.filter(pl.col('rescore_slice')==-1).drop(
            pl.selectors.matches('rescore.*')
        )
    )
    assert np.isclose(df_out2.filter(pl.col('rescore_slice')==-1)['rescore'], df_out3['rescore']).all()


@pytest.mark.df
def test_full_df_rescoring():
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


@pytest.mark.cli
def test_full_cli_parquet_rescoring():
    with tempfile.TemporaryDirectory(prefix='pytest_xirescore_') as tmpdirname:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        logger.info('Start full CSV rescoring test')
        logger.info(f'Write results to {tmpdirname}')

        options = {
            'input': {
                'columns': {
                    'features': [
                        'match_score',
                        'better_score',
                        'worse_score',
                        'useless_score_uni',
                        'useless_score_norm',
                        'conditional_score',
                    ],
                    'score': 'match_score'
                },
            },
            'rescoring': {
                'spectra_batch_size': 25_000,  # Rescore in 4 batches
            }
        }

        python_version = f'{sys.version_info.major}.{sys.version_info.minor}'
        result = subprocess.run(
            [
                f'python{python_version}',
                'xirescore',
                '-i', './tests/fixtures/test_data.parquet',
                '-o', f'{tmpdirname}/result.parquet',
                '-C', str(options),
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0, f"CLI command failed with error: {result.stderr}"
        assert os.path.exists(f'{tmpdirname}/result.parquet'), f"Output file was not created."
