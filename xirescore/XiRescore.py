"""Main module."""
import copy
import logging
import random
from collections.abc import Collection
from math import ceil

import numpy as np
import pandas as pd
import polars as pl
from deepmerge import Merger
from sklearn.base import BaseEstimator, TransformerMixin

from xirescore import readers
from xirescore import rescoring
from xirescore import train_data_selecting
from xirescore import training
from xirescore import writers
from xirescore._default_options import default_options
from xirescore.column_generating import generate as generate_columns
from xirescore.feature_extracting import get_features
from xirescore.feature_scaling import get_scaler
from xirescore.hyperparameter_optimizing import get_hyperparameters

options_merger = Merger(
    # pass in a list of tuple, with the
    # strategies you are looking to apply
    # to each type.
    [
        (dict, ["merge"]),
        (list, ["override"]),
        (set, ["override"])
    ],
    # next, choose the fallback strategies,
    # applied to all other types:
    ["override"],
    # finally, choose the strategies in
    # the case where the types conflict:
    ["override"]
)


class XiRescore:
    def __init__(self,
                 input_path,
                 output_path=None,
                 options=dict(),
                 logger=None,
                 loglevel=logging.DEBUG):
        """
        Initialize rescorer

        :param input_path: Path to input file/DB or an input DataFrame.
        :type input_path: str|DataFrame
        :param output_path: Path to the output file/DB or ``None`` if ``get_rescored_output()`` will be used.
        :type output_path: str, optional
        :param options: :ref:`options`
        :type options: dict, optional
        :param logger: Logger to be used. If ``None`` a new logger will be created.
        :type logger: Logger, optional
        :param loglevel: Log level to be used with new logger.
        :type loglevel: int, optional
        """
        # Apply override default options with user-supplied options
        self._options = copy.deepcopy(default_options)
        if 'model_params' in options.get('rescoring', dict()):
            # Discard default model_params if new ones are provided
            del self._options['rescoring']['model_params']
        self._options = options_merger.merge(
            self._options,
            options
        )

        # Set random seed
        seed = self._options['rescoring']['random_seed']
        self._true_random_seed = random.randint(0, 2**32-1)
        np.random.seed(seed)
        random.seed(seed)


        # Store input data path
        self._input = input_path

        if output_path is None:
            # Store output in new DataFrame if no path is given
            self._output = pl.DataFrame()
        else:
            self._output = output_path

        # Use supplied logger if present
        if logger is not None:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self._loglevel = loglevel
        self._true_random_ctr = 0

        self.train_df: pl.DataFrame
        """
        Data used for k-fold cross-validation.
        """
        self.splits: Collection[tuple[list, list]] = []
        """
        K-fold splits of model training. Kept to not rescore training samples with models they have been trained on.
        """
        self.models: list[BaseEstimator] = []
        """
        Trained models from the f-fold cross-validation.
        """
        self.scaler: TransformerMixin
        """
        Scaler for feature normalization.
        """
        self.train_features: list = []
        """
        Features extracted from training data.
        """

    def run(self) -> None:
        """
        Run training and rescoring of the input data and write to output
        """
        self._logger.info("Start full train and rescore run")
        self.train()
        self.rescore()

    def train(self,
              train_df: pl.DataFrame = None,
              splits: list[tuple[list, list]] = None):
        """
        Run training on input data or on the passed DataFrame if provided.

        :param train_df: Data to be used training instead of input data.
        :type train_df: DataFrame, optional
        :param splits: K-fold splits for manual ``train_df``.
        :type splits: Index, optional
        """
        self._logger.info('Start training')

        if train_df is None:
            self.train_df, self.scaler = train_data_selecting.select(
                self._input,
                self._options,
            )
        else:
            self.train_df = generate_columns(
                train_df,
                options=self._options,
                do_fdr=True,
                do_self_between=True
            )
            self.scaler = get_scaler(train_df, self._options)

        if splits is not None:
            self.splits = splits

        self.train_features = get_features(self.train_df, self._options)

        # Scale features
        train_df_transformed = self.train_df.clone()
        train_df_transformed[self.train_features] = self.scaler.transform(
            self.train_df[self.train_features]
        )

        self._logger.info("Perform hyperparameter optimization")
        model_params = get_hyperparameters(
            train_df=train_df_transformed,
            cols_features=self.train_features,
            splits=splits,
            options=self._options,
        )

        self._logger.info("Train models")
        self.models, self.splits = training.train(
            train_df=self.train_df,
            cols_features=self.train_features,
            clf_params=model_params,
            splits=splits,
            logger=self._logger,
            options=self._options,
        )

    def get_rescoring_state(self) -> dict:
        """
        Get state of the current instance to use it later to recreate identical instance.

        :returns: Models and k-fold slices
        :rtype: dict
        """
        return {
            'splits': self.splits,
            'models': self.models,
        }

    def _true_random(self, min_val=0, max_val=2**32-1):
        state = random.getstate()
        random.seed(self._true_random_seed+self._true_random_ctr)
        self._true_random_ctr += 1
        val = random.randint(min_val, max_val)
        random.setstate(state)
        return val

    def rescore(self) -> None:
        """
        Run rescoring on input data.
        """
        self._logger.info('Start rescoring')
        cols_spectra = self._options['input']['columns']['spectrum_id']
        spectra_batch_size = self._options['rescoring']['spectra_batch_size']

        # Read spectra list
        spectra = readers.read_spectra_ids(
            self._input,
            cols_spectra,
        )

        # Sort spectra
        spectra.sort()

        # Calculate number of batches
        n_batches = ceil(len(spectra)/spectra_batch_size)
        self._logger.info(f'Rescore in {n_batches} batches')

        # Iterate over spectra batches
        df_rescored = pl.DataFrame()
        for i_batch in range(n_batches):
            # Define batch borders
            spectra_range = spectra[
                i_batch*spectra_batch_size:(i_batch+1)*spectra_batch_size
            ]
            spectra_from = spectra_range[0]
            spectra_to = spectra_range[-1]
            self._logger.info(f'Start rescoring spectra batch {i_batch+1}/{n_batches} with `{spectra_from}` to `{spectra_to}`')

            # Read batch
            df_batch = readers.read_spectra_range(
                input=self._input,
                spectra_from=spectra_from,
                spectra_to=spectra_to,
                spectra_cols=cols_spectra,
                sequence_p2_col=self._options['input']['columns']['base_sequence_p2'],
                only_pairs=True,
            )
            self._logger.info(f'Batch contains {len(df_batch):,.0f} samples')
            self._logger.debug(f'Batch uses approx. {df_batch.estimated_size("mb"):,.2f}MB of RAM')

            # Rescore batch
            df_batch = self.rescore_df(df_batch)

            # Store collected matches
            self._logger.info('Write out batch')
            if type(self._output) is pl.DataFrame:
                df_rescored = pl.concat([
                    df_rescored,
                    df_batch
                ])
            else:
                writers.append_rescorings(
                    self._output,
                    df_batch,
                    options=self._options,
                    logger=self._logger,
                    random_seed=self._true_random()
                )

        # Keep rescored matches when no output is defined
        if type(self._output) is pl.DataFrame:
            self._output = df_rescored

    def rescore_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Rescore a DataFrame of CSMs.

        :param df: CSMs to be rescored
        :type df: DataFrame

        :return: Rescored CSMs
        :rtype: DataFrame
        """
        cols_spectra = self._options['input']['columns']['spectrum_id']
        col_rescore = self._options['output']['columns']['rescore']
        col_top_ranking = self._options['input']['columns']['top_ranking']
        max_jobs = self._options['rescoring']['max_jobs']
        apply_logit = self._options['rescoring']['logit_result']
        if self._options['input']['columns']['csm_id'] is None:
            col_csm = list(self.train_df.columns)
        else:
            col_csm = self._options['input']['columns']['csm_id']

        # Scale features
        df_scaled_features = pl.DataFrame(
            self.scaler.transform(
                df[self.train_features]
            ),
            schema=self.train_features
        )

        # Rescore DF
        df_scores = rescoring.rescore(
            models=self.models,
            df=df_scaled_features,
            rescore_col=col_rescore,
            apply_logit=apply_logit,
            max_cpu=max_jobs
        )

        self._logger.info('Merge new scores into original data')

        # Rescore training data only with test fold classifier
        self._logger.info('Reconstruct training data slices')
        cols_merge = list(set(col_csm+cols_spectra))
        df_slice = self.train_df.select(cols_merge)
        df_slice = df_slice.with_columns(
            pl.lit(-1).alias(f'{col_rescore}_slice')
        )
        for i, (_, idx_test) in enumerate(self.splits):
            # TODO move the slice assignment into `training()`
            df_slice = df_slice.with_columns(
                pl.when(
                    pl.int_range(len(df_slice)).is_in(idx_test)
                ).then(pl.lit(i)).otherwise(
                    pl.col(f'{col_rescore}_slice')
                ).alias(f'{col_rescore}_slice')
            )

        self._logger.info('Add merge columns to scores DataFrame')
        df_scores = pl.concat([
            df,
            df_scores,
        ], how="horizontal")

        self._logger.info('Merge slice info into batch')
        df_scores = df_slice.join(
            df_scores,
            on=cols_merge,
            how='right',
        )
        df_scores = df_scores.with_columns(
            pl.col(f'{col_rescore}_slice').fill_null(-1)
        )

        self._logger.info('Pick the correct score')
        score_ser = df_scores.select(
            pl.col(f'^{col_rescore}_[0-9]+$')
        ).cast(pl.List(pl.Float64)).select(
            pl.concat_list(pl.col('*'))
        ).to_series()
        df_scores = df_scores.with_columns(
            scores_list=score_ser
        ).with_columns(
            pl.when(
                pl.col(f'{col_rescore}_slice') > -1
            ).then(
                pl.col('scores_list').list.get(
                    pl.col(f'{col_rescore}_slice')
                )
            ).otherwise(
                pl.col(col_rescore)
            ).alias(col_rescore)
        ).drop('scores_list')

        # Calculate top_ranking
        self._logger.info('Calculate top ranking scores')
        df_top_rank = df_scores.group_by(cols_spectra).agg(
            pl.max(f'{col_rescore}').alias(f'{col_rescore}_max'),
            pl.max(f'{col_rescore}').alias(f'{col_rescore}_min'),
        )
        df_scores = df_scores.join(
            df_top_rank,
            on=list(cols_spectra)
        )
        df_scores = df_scores.with_columns(
            pl.col(col_rescore).neg().rank('dense').alias(f'{col_rescore}_rank')
        ).with_columns(
            (pl.col(f'{col_rescore}_rank') == 1).alias(f'{col_rescore}_{col_top_ranking}')
        )

        return df_scores

    def get_rescored_output(self) -> pl.DataFrame:
        """
        Get the rescoring results when no output was defined

        :returns: Rescoring results
        :rtype: DataFrame
        """
        if type(self._output) is pl.DataFrame:
            return self._output
        else:
            raise XiRescoreError('Not available for file output.')


def _select_right_score(row, col_rescore):
    n_slice = int(row[f"{col_rescore}_slice"])
    return row[f'{col_rescore}_{n_slice}']


class XiRescoreError(Exception):
    """Custom exception for train data selection errors."""
    pass
