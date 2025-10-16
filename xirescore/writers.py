"""
Writers for data outputs
"""
from pathlib import Path
import random
import os
from typing import Union

import fastparquet
import pandas as pd
import polars as pl

from xirescore.readers import get_source_type
from xirescore.df_serializing import serialize_columns


def append_rescorings(output, df: pd.DataFrame, schema_overrides: dict = {}):
    output_type = get_source_type(output)
    df = df.cast({
        k: v
        for k, v in schema_overrides.items()
        if k in df.columns
    })
    if output_type == 'csv':
        append_csv(output, df)
    if output_type == 'tsv':
        append_csv(output, df, sep='\t')
    if output_type == 'parquet':
        append_parquet(output, df)


def append_parquet(output, df: Union[pd.DataFrame, pl.DataFrame], compression='GZIP'):
    if type(df) is pl.DataFrame:
        df = df.to_pandas()
    else:
        df = serialize_columns(df)
    fastparquet.write(
        output,
        data=df,
        compression=compression,
        write_index=False,
        append=Path(output).is_file(),
    )


def append_csv(output, df: Union[pd.DataFrame, pl.DataFrame], sep=','):
    if type(df) is pl.DataFrame:
        df = df.to_pandas()
    else:
        df = serialize_columns(df)
    df.to_csv(
        output,
        mode='a',
        header=not os.path.isfile(output),
        sep=sep
    )
