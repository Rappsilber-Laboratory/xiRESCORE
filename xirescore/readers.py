"""
Readers for data inputs
"""
import random
from math import ceil
from typing import Union
from collections.abc import Sequence

import pandas as pd
import polars as pl
from fastparquet import ParquetFile as FPParquetFile


def read_spectra_ids(path, spectra_cols=None, schema_overrides={}) -> list[int]:
    if type(path) in [pd.DataFrame, pl.DataFrame]:
        df = pl.DataFrame(path)
        return df.select(pl.struct(spectra_cols).hash())\
            .unique() \
            .to_series().to_list()

    file_type = get_source_type(path)

    if spectra_cols is None:
        raise ValueError('Filetype {file_type} requires parameter `spectra_cols`!')

    if file_type == 'csv':
        return pl.scan_csv(
                path,
                ignore_errors=True,
                null_values=['∞', '-∞'],
                schema_overrides=schema_overrides,
            ).select(pl.struct(spectra_cols).hash()) \
            .unique() \
            .collect().to_series().to_list()
    if file_type == 'tsv':
        return pl.scan_csv(
                path,
                separator='\t',
                ignore_errors=True,
                null_values=['∞', '-∞'],
                schema_overrides=schema_overrides,
            ).select(pl.struct(spectra_cols).hash()) \
            .unique() \
            .collect().to_series().to_list()
    if file_type == 'parquet':
        return pl.scan_parquet(path) \
            .select(pl.struct(spectra_cols).hash()) \
            .unique() \
            .collect().to_series().to_list()


def read_spectra_range(input: Union[str, pd.DataFrame],
                       spectra_from: int,
                       spectra_to: int,
                       spectra_cols: Sequence = None,
                       sequence_p2_col='sequence_p2',
                       only_pairs=True,
                       schema_overrides={}):
    # Convert pandas to polars
    if type(input) is pd.DataFrame:
        input = pl.DataFrame(input)
    # Handle input DF
    if type(input) is pl.DataFrame:
        filters = (
            (pl.struct(spectra_cols).hash() >= spectra_from) &
            (pl.struct(spectra_cols).hash() <= spectra_to)
        )
        if only_pairs:
            filters &= pl.col(sequence_p2_col).is_not_null()
            filters &= pl.col(sequence_p2_col) != ''
        return input.filter(filters)
    # Handle input path
    file_type = get_source_type(input)
    if file_type == 'csv':
        return read_spectra_range_csv(
            input,
            spectra_from,
            spectra_to,
            spectra_cols=spectra_cols,
            sequence_p2_col=sequence_p2_col,
            only_pairs=only_pairs,
            schema_overrides=schema_overrides,
        )
    if file_type == 'tsv':
        return read_spectra_range_csv(
            input,
            spectra_from,
            spectra_to,
            sep='\t',
            spectra_cols=spectra_cols,
            sequence_p2_col=sequence_p2_col,
            only_pairs=only_pairs,
            schema_overrides=schema_overrides,
        )
    if file_type == 'parquet':
        return read_spectra_range_parquet(
            input,
            spectra_from,
            spectra_to,
            spectra_cols=spectra_cols,
            sequence_p2_col=sequence_p2_col,
            only_pairs=only_pairs,
        )


def read_spectra_range_parquet(path,
                               spectra_from,
                               spectra_to,
                               spectra_cols: Sequence,
                               sequence_p2_col='sequence_p2',
                               only_pairs=True):
    # Filters for spectrum columns
    df = pl.scan_parquet(path)
    # Generate filters
    filters = (
        (pl.struct(spectra_cols).hash() >= spectra_from) &
        (pl.struct(spectra_cols).hash() <= spectra_to)
    )
    # Filter out linear matches
    if only_pairs:
        filters &= pl.col(sequence_p2_col).is_not_null()
        filters &= pl.col(sequence_p2_col) != ''
    return df.filter(filters).collect()


def read_spectra_range_csv(path,
                           spectra_from,
                           spectra_to,
                           spectra_cols: Sequence,
                           sequence_p2_col='sequence_p2',
                           only_pairs=True,
                           sep=',',
                           schema_overrides={}):
    # Filters for spectrum columns
    df = pl.scan_csv(
        path,
        separator=sep,
        ignore_errors=True,
        null_values=['∞', '-∞'],
        schema_overrides=schema_overrides,
    )
    # Generate filters
    filters = (
        (pl.struct(spectra_cols).hash() >= spectra_from) &
        (pl.struct(spectra_cols).hash() <= spectra_to)
    )
    # Filter out linear matches
    if only_pairs:
        filters &= pl.col(sequence_p2_col).is_not_null()
        filters &= pl.col(sequence_p2_col) != ''
    return df.filter(filters).collect()


def get_source_type(path: str):
    if path.lower().endswith('.parquet') or path.lower().endswith('.parquet/'):
        return 'parquet'
    if path.lower().endswith('.tsv') or path.lower().endswith('.tab'):
        return 'tsv'
    if path.lower().endswith('.csv'):
        return 'csv'
    if len(path.split('.')) > 2:
        ext2 = path.split('.')[-2].lower()
        if ext2 == 'csv':
            return 'csv'
        if ext2 == 'tab' or ext2 == 'tsv':
            return 'tsv'
    raise ValueError(f'Unknown file type of {path}')


def read_sample(input_data,
                sample=1_000_000,
                top_ranking_col='top_ranking',
                sequence_p2_col='sequence_p2',
                only_top_ranking=False,
                only_pairs=True,
                schema_overrides={}) -> pl.DataFrame:
    if type(input_data) is pd.DataFrame:
        input_data = pl.DataFrame(input_data)
    if type(input_data) is pl.DataFrame:
        scan_filter = pl.lit(True)
        if only_top_ranking and top_ranking_col in input_data.collect_schema().names():
            scan_filter &= pl.col(top_ranking_col)
        if only_pairs:
            scan_filter &= pl.col(sequence_p2_col).is_not_null()
            scan_filter &= pl.col(sequence_p2_col) != ''
        sample_min = min(
            len(input_data.filter(scan_filter)),
            sample
        )
        return input_data.filter(scan_filter).sample(sample_min)
    file_type = get_source_type(input_data)
    if file_type == 'csv':
        return read_sample_csv(
            input_data,
            sample=sample,
            top_ranking_col=top_ranking_col,
            sequence_p2_col=sequence_p2_col,
            only_top_ranking=only_top_ranking,
            only_pairs=only_pairs,
            schema_overrides=schema_overrides,
        )
    if file_type == 'tsv':
        return read_sample_csv(
            input_data,
            sep='\t',
            sample=sample,
            top_ranking_col=top_ranking_col,
            only_top_ranking=only_top_ranking,
            sequence_p2_col=sequence_p2_col,
            only_pairs=only_pairs,
            schema_overrides=schema_overrides,
        )
    if file_type == 'parquet':
        return read_sample_parquet(
            input_data,
            sample=sample,
            sequence_p2_col=sequence_p2_col,
            only_top_ranking=only_top_ranking,
            top_ranking_col=top_ranking_col,
            only_pairs=only_pairs,
        )


def read_sample_parquet(path: str,
                        sample: int,
                        top_ranking_col='top_ranking',
                        sequence_p2_col='sequence_p2',
                        batch_size=500_000,
                        only_top_ranking=False,
                        only_pairs=True,
                        random_state=random.randint(0, 2**32-1)):
    df_scan = pl.scan_parquet(path)

    scan_filter = pl.lit(True)
    if only_top_ranking and top_ranking_col in df_scan.columns:
        scan_filter &= pl.col(top_ranking_col)
    if only_pairs:
        scan_filter &= pl.col(sequence_p2_col).is_not_null()
        scan_filter &= pl.col(sequence_p2_col) != ''

    df_scan = df_scan.filter(scan_filter)

    n_total = df_scan.select(pl.count()).collect().item()
    every_nth = ceil(n_total / sample)

    return df_scan.gather_every(every_nth).collect()


def read_sample_csv(path,
                    sample,
                    sep=',',
                    top_ranking_col='top_ranking',
                    sequence_p2_col='sequence_p2',
                    only_pairs=True,
                    only_top_ranking=False,
                    schema_overrides={}):
    df_scan = pl.scan_csv(
        path,
        separator=sep,
        ignore_errors=True,
        null_values=['∞', '-∞'],
        schema_overrides=schema_overrides
    )

    scan_filter = pl.lit(True)
    if only_top_ranking and top_ranking_col in df_scan.collect_schema().names():
        scan_filter &= pl.col(top_ranking_col)
    if only_pairs:
        scan_filter &= pl.col(sequence_p2_col).is_not_null()
        scan_filter &= pl.col(sequence_p2_col) != ''

    df_scan = df_scan.filter(scan_filter)
    n_total = df_scan.select(pl.count()).collect().item()
    every_nth = ceil(n_total / sample)

    return df_scan.gather_every(every_nth).collect()
