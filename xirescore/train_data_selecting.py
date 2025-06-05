import logging

import polars as pl
import numpy as np
from sklearn import impute

from xirescore import readers
from xirescore.column_generating import generate as generate_columns
from xirescore.feature_scaling import get_transformers


logger = logging.getLogger(__name__)

def select(input_data, options):
    """
    Select training data for Crosslink MS Machine Learning based on specified options.

    Parameters:
    - input_data: The input data for selection.
    - options: Dictionary containing various configuration options for the selection process.
    - logger: Logger instance for logging information and debugging.

    Returns:
    - A pandas DataFrame containing the selected training data.
    """

    # Extract options
    selection_mode = options['rescoring']['train_selection_mode']
    train_unique = options['rescoring']['train_unique_csms']
    train_size_max = options['rescoring']['train_size_max']
    top_sample_size = options['rescoring']['top_sample_size']
    top_ranking_col = options['input']['columns']['top_ranking']
    col_sequence_p1 = options['input']['columns']['base_sequence_p1']
    col_sequence_p2 = options['input']['columns']['base_sequence_p2']
    col_base_sequence_p2 = options['input']['columns']['base_sequence_p2']
    seed = options['rescoring']['random_seed']
    col_self_between = options['input']['columns']['self_between']
    col_fdr = options['input']['columns']['fdr']
    col_native_score = options['input']['columns']['score']
    col_target = options['input']['columns']['is_tt']
    col_link_pos_p1 = options['input']['columns']['link_pos_p1']
    col_link_pos_p2 = options['input']['columns']['link_pos_p2']
    col_charge = options['input']['columns']['charge']
    fdr_cutoff = options['rescoring']['train_fdr_threshold']
    val_self = options['input']['constants']['self']

    # Generate schema overrides
    float_cols = options['input']['columns']['features']
    float_cols.append(
        options['input']['columns']['score']
    )
    schema_overrides = {
        c: pl.Float64
        for c in float_cols
    }

    # Read input data
    df = readers.read_sample(
        input_data,
        sample=top_sample_size,
        top_ranking_col=top_ranking_col,
        sequence_p2_col=col_base_sequence_p2,
        only_top_ranking=True,
        only_pairs=True,
        schema_overrides=schema_overrides,
    )
    logger.debug(f'Fetched {len(df)} top ranking samples')

    if train_unique:
        unique_cols = [
            col_sequence_p1,
            col_sequence_p2,
            col_link_pos_p1,
            col_link_pos_p2,
            col_charge,
        ]
        df = df.sort(col_native_score, descending=True).unique(subset=unique_cols, keep='first')

    # Generate needed columns
    df = generate_columns(df, options=options, do_fdr=True, do_self_between=True)

    # Get scaler
    imputer, scaler, features = get_transformers(df, options)

    # Selection mode: self-targets-all-decoys
    logger.info(f'Use selection mode {selection_mode}')
    if selection_mode == 'self-targets-all-decoys':
        # Create filters
        filter_self = pl.col(col_self_between) == val_self
        filter_fdr = pl.col(col_fdr) <= fdr_cutoff
        filter_target = pl.col(col_target)

        # Max target size
        target_max = int(train_size_max / 2)

        # Get self targets
        train_self_targets = df.filter(filter_fdr & filter_target & filter_self)
        if len(train_self_targets) > target_max:
            train_self_targets = train_self_targets.sample(target_max, seed=seed)
        logger.info(f'Taking {len(train_self_targets)} self targets below {fdr_cutoff} FDR')

        # Get between targets
        train_between_targets = df.filter(filter_fdr & filter_target & ~filter_self)
        sample_min = min(
            len(train_between_targets),
            int(train_size_max/2)-len(train_self_targets),
        )
        train_between_targets = train_between_targets.sample(sample_min, seed=seed)
        logger.info(f'Taking {len(train_between_targets)} between targets below {fdr_cutoff} FDR')

        # Get self decoy-x
        train_self_decoys = df.filter(filter_self & ~filter_target)
        sample_min = min(
            len(train_self_decoys),
            int(train_size_max/4)
        )
        train_self_decoys = train_self_decoys.sample(sample_min, seed=seed)
        logger.info(f'Taking {len(train_self_decoys)} self decoys.')

        # Get between decoy-x
        train_between_decoys = df.filter((~filter_self) & (~filter_target))
        sample_min = min(
            len(train_between_decoys),
            int(train_size_max/2)-len(train_self_decoys),
        )
        train_between_decoys = train_between_decoys.sample(sample_min, seed=seed)
        logger.info(f'Taking {len(train_between_decoys)} between decoys.')

        train_data_df = pl.concat([
            train_self_targets,
            train_between_targets,
            train_self_decoys,
            train_between_decoys
        ])

    # Selection mode: self-targets-capped-decoys
    elif selection_mode == 'self-targets-capped-decoys':
        # Create filters
        filter_self = pl.col(col_self_between) == val_self
        filter_fdr = pl.col(col_fdr) <= fdr_cutoff
        filter_target = pl.col(col_target)

        # Max target size
        target_max = int(train_size_max / 2)

        # Get self targets
        train_self_targets = df.filter(filter_fdr & filter_target & filter_self)
        if len(train_self_targets) > target_max:
            train_self_targets = train_self_targets.sample(target_max, seed=seed)
        logger.info(f'Taking {len(train_self_targets)} self targets below {fdr_cutoff} FDR')

        # Get between targets
        train_between_targets = df.filter(filter_fdr & filter_target & ~filter_self)
        sample_min = min(
            len(train_between_targets),
            int(train_size_max/2)-len(train_self_targets),
        )
        train_between_targets = train_between_targets.sample(sample_min, seed=seed)
        logger.info(f'Taking {len(train_between_targets)} between targets below {fdr_cutoff} FDR')

        # Get capped decoy-x
        all_target = df.filter(filter_target)
        all_decoy = df.filter(~filter_target)
        _, hist_bins = np.histogram(df[col_native_score].to_numpy(), bins=1_000)
        hist_tt, _ = np.histogram(all_target[col_native_score].to_numpy(), bins=hist_bins)
        hist_dx, _ = np.histogram(all_decoy[col_native_score].to_numpy(), bins=hist_bins)
        hist_dx_capped = np.minimum(hist_dx, hist_tt)

        # Number of Dx to aim for
        n_decoy = min([
            hist_dx_capped.sum(),
            train_size_max/2
        ])

        # Scale decoy-x histogram
        dx_scale_fact = min(
            1,
            n_decoy / hist_dx_capped.sum(),
        )
        hist_dx_scaled = (hist_dx_capped * dx_scale_fact).round().astype(int)

        train_decoys = pl.DataFrame(schema=df.schema)
        for i, n in enumerate(hist_dx_scaled):
            if n == 0:
                continue
            score_min = hist_bins[i]
            score_max = hist_bins[i + 1]
            bins_samples = all_decoy.filter(
                (pl.col(col_native_score) >= score_min) & (pl.col(col_native_score) < score_max)
            )
            train_decoys = pl.concat(
                [
                    train_decoys,
                    bins_samples.sample(n=n, seed=seed)
                ]
            )

        logger.info(f'Taking {len(train_decoys)} decoys.')

        train_data_df = pl.concat([
            train_self_targets,
            train_between_targets,
            train_decoys]
        )
    else:
        raise TrainDataError(f"Unknown train data selection mode: {selection_mode}.")

    return train_data_df, imputer, scaler, features


class TrainDataError(Exception):
    """Custom exception for train data selection errors."""
    pass
