import logging
import polars as pl

logger = logging.getLogger(__name__)

def validate(df: pl.DataFrame, options):
    # Validate that all feature columns are present
    for fcol in options['input']['columns']['features']:
        if fcol not in df.columns:
            logger.warning(f"Feature column `{fcol}` not in input DataFrame.")

    # Validate that at least score or fdr or training_flag column is present
    col_score = options['input']['columns']['score']
    if col_score not in df.columns:
        raise XiRescoreInputError(f"Score column `{col_score}` was not found.")

    # Validate that sequence columns are present
    col_seq1 = options['input']['columns']['base_sequence_p1']
    col_seq2 = options['input']['columns']['base_sequence_p2']
    if col_seq1 not in df.columns:
        raise XiRescoreInputError(f"Missing sequence column `{col_seq1}`.")
    if col_seq2 not in df.columns:
        raise XiRescoreInputError(f"Missing sequence column `{col_seq2}`.")

    # Validate that target column is present
    col_target = options['input']['columns']['is_tt']
    if col_target not in df.columns:
        raise XiRescoreInputError(f"Missing target column `{col_target}`.")


class XiRescoreInputError(Exception):
    pass
