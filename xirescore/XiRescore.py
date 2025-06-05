"""Main module."""
import copy
import logging
import random
from collections.abc import Collection
from math import ceil

import numpy as np
from networkx.algorithms.bipartite.cluster import modes
from sklearn.decomposition import PCA
import polars as pl
from deepmerge import Merger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance

from xirescore import readers
from xirescore import rescoring
from xirescore import train_data_selecting
from xirescore import training
from xirescore import writers
from xirescore._default_options import default_options
from xirescore.column_generating import generate as generate_columns
from xirescore.feature_extracting import get_features
from xirescore.feature_scaling import get_transformers
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

logger = logging.getLogger(__name__)


class XiRescore:
    def __init__(self,
                 input_path,
                 output_path=None,
                 options=dict()):
        """
        Initialize rescorer

        :param input_path: Path to input file/DB or an input DataFrame.
        :type input_path: str|DataFrame
        :param output_path: Path to the output file/DB or ``None`` if ``get_rescored_output()`` will be used.
        :type output_path: str, optional
        :param options: :ref:`options`
        :type options: dict, optional
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
        self.scaler: TransformerMixin = None
        """
        PCA for feature decorrelation
        """
        self.pca: PCA = None
        """
        Scaler for feature normalization.
        """
        self.train_features: list = []
        """
        Features extracted from training data.
        """
        self.imputer: TransformerMixin = None
        """
        Imputer for missing values.
        """

    def run(self) -> None:
        """
        Run training and rescoring of the input data and write to output
        """
        logger.info("Start full train and rescore run")
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
        logger.info('Start training')

        if train_df is None:
            self.train_df, self.imputer, self.scaler, self.train_features = train_data_selecting.select(
                self._input,
                self._options,
            )
        else:
            self.train_df = train_df
            self.train_df = generate_columns(
                self.train_df,
                options=self._options,
                do_fdr=True,
                do_self_between=True
            )
            self.imputer, self.scaler, self.train_features = get_transformers(train_df, self._options)

        if splits is not None:
            self.splits = splits

        # Scale features
        train_df_transformed = self.train_df.clone()
        if self.imputer is not None:
            train_df_transformed[self.train_features] = self.imputer.transform(
                train_df_transformed[self.train_features]
            )
        train_df_transformed[self.train_features] = self.scaler.transform(
            train_df_transformed[self.train_features]
        )
        train_df_transformed = train_df_transformed.fill_nan(0)

        passed_feaures=self.train_features
        if self._options['rescoring']['pca_n_components'] is not None:
            self.pca = PCA(n_components=self._options['rescoring']['pca_n_components'])
            pca_features = self.pca.fit_transform(train_df_transformed[self.train_features])
            passed_feaures = [f'_pca_feature_{i}' for i  in range(pca_features.shape[1])]
            logger.info(f"PCA returned {len(passed_feaures)} features")
            train_df_transformed[passed_feaures] = pca_features


        logger.info("Perform hyperparameter optimization")
        model_params = get_hyperparameters(
            train_df=train_df_transformed,
            cols_features=passed_feaures,
            splits=splits,
            options=self._options,
        )

        logger.info("Train models")
        self.models, self.splits = training.train(
            train_df=train_df_transformed,
            cols_features=passed_feaures,
            clf_params=model_params,
            splits=splits,
            options=self._options,
        )

    def get_rescoring_state(self) -> dict:
        """
        Get state of the current instance to use it later to recreate identical instance.

        :returns: Models and k-fold slices
        :rtype: dict
        """
        cols_csm = self._options['input']['columns']['csm_id']
        cols_spectra = self._options['input']['columns']['spectrum_id']
        train_cols = list(set(cols_csm+cols_spectra))

        return {
            'options': self._options,
            'pca': self.pca,
            'imputer': self.imputer,
            'scaler': self.scaler,
            'training_data': self.train_df.select(train_cols),
            'train_features': self.train_features,
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
        logger.info('Start rescoring')
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
        logger.info(f'Rescore in {n_batches} batches')

        # Iterate over spectra batches
        df_rescored = pl.DataFrame()
        for i_batch in range(n_batches):
            # Define batch borders
            spectra_range = spectra[
                i_batch*spectra_batch_size:(i_batch+1)*spectra_batch_size
            ]
            spectra_from = spectra_range[0]
            spectra_to = spectra_range[-1]
            logger.info(f'Start rescoring spectra batch {i_batch+1}/{n_batches} with `{spectra_from}` to `{spectra_to}`')

            # Generate schema overrides
            float_cols = self._options['input']['columns']['features']
            float_cols.append(
                self._options['input']['columns']['score']
            )
            schema_overrides = {
                c: pl.Float64
                for c in float_cols
            }

            # Read batch
            df_batch = readers.read_spectra_range(
                input=self._input,
                spectra_from=spectra_from,
                spectra_to=spectra_to,
                spectra_cols=cols_spectra,
                sequence_p2_col=self._options['input']['columns']['base_sequence_p2'],
                only_pairs=True,
                schema_overrides=schema_overrides
            )
            logger.info(f'Batch contains {len(df_batch):,.0f} samples')
            logger.debug(f'Batch uses approx. {df_batch.estimated_size("mb"):,.2f}MB of RAM')

            # Rescore batch
            df_batch = self.rescore_df(df_batch)

            # Store collected matches
            logger.info('Write out batch')
            if type(self._output) is pl.DataFrame:
                df_rescored = pl.concat([
                    df_rescored,
                    df_batch
                ])
            else:
                writers.append_rescorings(
                    self._output,
                    df_batch
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

        # Impute missing feature values
        df_features = df[self.train_features]
        if self.imputer is not None:
            df_features = self.imputer.transform(df_features)

        # Scale features
        df_scaled_features = self.scaler.transform(df_features)
        df_scaled_features = np.nan_to_num(df_scaled_features)
        passed_feaures = self.train_features
        if self.pca is not None:
            df_scaled_features = self.pca.transform(df_scaled_features)
            passed_feaures = [f'_pca_feature_{i}' for i in range(df_scaled_features.shape[1])]
        df_scaled_features = pl.DataFrame(
            df_scaled_features,
            schema=passed_feaures
        )

        # Rescore DF
        df_scores = rescoring.rescore(
            models=self.models,
            df=df_scaled_features,
            rescore_col=col_rescore,
            apply_logit=apply_logit,
            max_cpu=max_jobs
        )

        logger.info('Merge new scores into original data')

        # Rescore training data only with test fold classifier
        logger.info('Reconstruct training data slices')
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

        logger.info('Add merge columns to scores DataFrame')
        df_scores = pl.concat([
            df,
            df_scores,
        ], how="horizontal")

        logger.info('Merge slice info into batch')
        df_scores = df_scores.join(
            df_slice,
            on=cols_merge,
            how='left',
            maintain_order='left'
        )
        df_scores = df_scores.with_columns(
            pl.col(f'{col_rescore}_slice').fill_null(-1)
        )

        logger.info('Pick the correct score')
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
        logger.info('Calculate top ranking scores')
        df_top_rank = df_scores.group_by(cols_spectra).agg(
            pl.max(f'{col_rescore}').alias(f'{col_rescore}_max'),
            pl.min(f'{col_rescore}').alias(f'{col_rescore}_min'),
            pl.col(col_rescore).neg().rank('dense').alias(f'{col_rescore}_rank'),
            pl.col(col_rescore),
        ).explode([
            col_rescore,
            f'{col_rescore}_rank'
        ]).unique()
        df_scores = df_scores.join(
            df_top_rank,
            on=list(cols_spectra)+[col_rescore],
            maintain_order='left'
        )
        df_scores = df_scores.with_columns(
            (
                pl.col(col_rescore) == pl.col(f'{col_rescore}_max')
            ).alias(f'{col_rescore}_{col_top_ranking}')
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

    def get_feature_importance(self) -> np.array:
        """
        Get the feature importances.
        """
        if not self.models:
            raise XiRescoreError('Models have not been trained.')
        if self.train_df is None:
            raise XiRescoreError('No training data defined.')

        # Check how to calculate feature importance
        importances = []
        for model in self.models:
            if hasattr(model, 'coef_'):
                importances.append(self._get_feature_importance_linear(model))
            elif hasattr(model, 'feature_importances_'):
                importances.append(self._get_feature_importance_tree(model))
            else:
                importances.append(self._get_feature_importance_permutation(model))

        if self.pca is not None:
            loadings = self.pca.components_.T
            orig_importances = []
            for importances_i in importances:
                orig_importances.append(loadings @ importances_i)
            importances = np.array(orig_importances)
        else:
            importances = np.array(importances)

        return importances

    def _get_feature_importance_linear(self, model) -> np.array:
        coef = model.coef_
        if coef.ndim == 1:
            importance = coef
        elif coef.ndim == 2:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            raise ValueError(f"Unexpected coef_ shape: {coef.shape}")
        return importance

    def _get_feature_importance_tree(self, model) -> np.array:
        return model.feature_importances_

    def _get_feature_importance_permutation(self, model) -> np.array:
        features = self.train_df.select(self.train_features)
        if self.pca is not None:
            features = self.pca.transform(features)
        features = self.scaler.transform(features)
        labels = self.train_df[self._options['input']['columns']['is_tt']]
        result = permutation_importance(
            model,
            features,
            labels,
            n_repeats=10,
            random_state=self._options['rescoring']['random_seed'],
            n_jobs=self._options['rescoring']['max_jobs']
        )
        return result.importances_mean


def _select_right_score(row, col_rescore):
    n_slice = int(row[f"{col_rescore}_slice"])
    return row[f'{col_rescore}_{n_slice}']


class XiRescoreError(Exception):
    """Custom exception for train data selection errors."""
    pass
