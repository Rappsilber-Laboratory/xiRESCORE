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
                },
                'schema_overrides': {
                    # Random type conversions
                    'worse_score': 'float',
                    'useless_score_uni': 'int',
                    'useless_score_norm': 'bool',
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
