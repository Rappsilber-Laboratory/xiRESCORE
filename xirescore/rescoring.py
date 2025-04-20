import multiprocess as mp
from math import ceil
import logging

import numpy as np
import polars as pl
import scipy

from xirescore import async_result_resolving


logger = logging.getLogger(__name__)

def rescore(models,
            df,
            rescore_col,
            apply_logit=False,
            max_cpu=-1):
    if not isinstance(df, pl.DataFrame):
        df: pl.DataFrame = pl.DataFrame(df)
    n_procs = max_cpu
    if n_procs < 1:
        n_procs = int(mp.cpu_count() - 1)
    n_models = len(models)
    n_dataslices = ceil(n_procs/n_models)
    slice_size = ceil(len(df) / n_dataslices)
    logger.debug(f'Split {len(df):,.0f} samples in {n_dataslices} slices of {slice_size}')

    # Slice input data for multiprocessing
    dataslices = [
        df[
            i*slice_size:(i+1)*slice_size
        ]
        for i in range(n_dataslices)
    ]

    # Apply each classifier to each data slice
    with mp.get_context("spawn").Pool(n_procs) as pool:
        async_results = []
        for slice in dataslices:
            for clf in models:
                if "Perceptron" in str(clf):
                    async_results.append(
                        pool.apply_async(
                            clf.decision_function,
                            (slice,),
                        )
                    )
                elif "tensorflow" not in str(clf):
                    async_results.append(
                        pool.apply_async(
                            clf.predict_proba,
                            (slice,),
                        )
                    )
                else:
                    async_results.append(
                        pool.apply_async(
                            clf.predict_proba,
                            (slice,),
                        )
                    )

        sync_results = async_result_resolving.resolve(async_results)
        slice_results: list = [
            # Join slices into a single array of all model predictions each
            np.array([
                sync_results[
                    (slice_i*n_models)+model_i
                ][:, 1]  # Take only prediction for class 1 (aka target)
                for model_i in range(n_models)
            ])
            for slice_i in range(n_dataslices)
        ]

    # Apply logit
    if apply_logit:
        for i, sr in enumerate(slice_results):
            slice_results[i] = scipy.special.logit(sr)

    # Init result DF
    np_rescore = np.zeros((len(df), 0))

    # Fill result DF
    for i_m, model in enumerate(models):
        rescores_m = [
            slice_results[i_s][i_m]
            for i_s in range(n_dataslices)
        ]
        np_rescore = np.hstack([
            np_rescore,
            np.concatenate(rescores_m).reshape(len(df), 1)
        ])

    df_rescore = pl.DataFrame(
        np_rescore,
        schema=[(f'{rescore_col}_{i_m}', pl.Float64) for i_m, _ in enumerate(models)],
    )

    # Calculate mean score and standard deviation
    df_rescore = df_rescore.with_columns(
        df_rescore.mean_horizontal().alias(rescore_col),
        df_rescore.transpose().std().transpose().to_series().alias(f'{rescore_col}_std'),
    )

    return df_rescore
