import polars as pl
from xifdr import fdr

def generate(df: pl.DataFrame, options: dict, do_self_between=False, do_fdr=False) -> pl.DataFrame:
    input_cols = options['input']['columns']
    consts = options['input']['constants']
    cols_spectra = input_cols['spectrum_id']
    col_score = input_cols['score']
    # Generate base sequences
    if input_cols['base_sequence_p1'] not in df.columns:
        # Remove modifications
        df = df.with_columns(
            pl.col(input_cols['sequence_p1'])\
                .str.replace_all('\\(.*\\)', '')\
                .str.replace_all('\\[.*\\]', '')\
                .str.replace_all('[^A-Z]', '')\
                .alias(input_cols['base_sequence_p1'])
        )
    if input_cols['base_sequence_p2'] not in df.columns:
        # Remove modifications
        df = df.with_columns(
            pl.col(input_cols['sequence_p2'])\
                .str.replace_all('\\(.*\\)', '')\
                .str.replace_all('\\[.*\\]', '')\
                .str.replace_all('[^A-Z]', '')\
                .alias(input_cols['base_sequence_p2'])
        )
    # Generate top_ranking
    if input_cols['top_ranking'] not in df.columns:
        df_max = df.group_by(cols_spectra).agg(
            pl.col(col_score).max().alias(f'{col_score}_max')
        )
        df = df.join(
            df_max,
            on=list(cols_spectra)
        )
        df = df.with_columns(
            top_ranking=pl.col(col_score) == pl.col(f'{col_score}_max')
        )
    # Generate decoy_p1/2
    ordered_decoy_class = (consts['td_class'] is not None) and (consts['dt_class'] is not None)
    if input_cols['decoy_class'] in df.columns and ordered_decoy_class:
        df = df.with_columns(
            (
                pl.col(input_cols['decoy_class']).is_in([consts['tt_class'], consts['td_class']])
            ).alias(input_cols['decoy_p1']),
            (
                pl.col(input_cols['decoy_class']).is_in([consts['dd_class'], consts['dt_class']])
            ).alias(input_cols['decoy_p2']),
        )
    # Generate decoy_class column from decoy_p1 and decoy_p2
    if input_cols['decoy_class'] not in df.columns:
        tt_expr = pl.col(input_cols['decoy_p1']).not_() & pl.col(input_cols['decoy_p2']).not_()
        dd_expr = pl.col(input_cols['decoy_p1']) & pl.col(input_cols['decoy_p2'])
        df = df.with_columns(
            pl.when(tt_expr).then(pl.lit(consts['tt_class']))\
                .when(dd_expr).then(pl.lit(consts['dd_class']))\
                .otherwise(pl.lit(consts['td_class']))\
                .alias(input_cols['decoy_class'])
        )
    # Generate target column from decoy_class
    if input_cols['is_tt'] not in df.columns:
        df = df.with_columns(
            (pl.col(input_cols['decoy_class']) == consts['tt_class']).alias(input_cols['is_tt'])
        )
    # Calculte self_between from protein_p1, and protein_p2
    if do_self_between and input_cols['self_between'] not in df.columns:
        protein_p1_list = pl.col(input_cols['protein_p1'])
        protein_p2_list = pl.col(input_cols['protein_p2'])
        if type(df[input_cols['protein_p1']].dtype) is pl.String:
            protein_p1_list = protein_p1_list.str.split(';')
        if type(df[input_cols['protein_p2']].dtype) is pl.String:
            protein_p2_list = protein_p2_list.str.split(';')
        protein_p1_list = protein_p1_list.list.eval(
            pl.element().str.replace_all(options['input']['constants']['decoy_adjunct'], '')
        )
        protein_p2_list = protein_p2_list.list.eval(
            pl.element().str.replace_all(options['input']['constants']['decoy_adjunct'], '')
        )
        overlap_expr = protein_p1_list.list.set_intersection(protein_p2_list)
        df = df.with_columns(
            fdr_group = pl.when(
                overlap_expr.list.len() == 0
            ).then(
                pl.lit('between')
            ).otherwise(
                pl.lit('self')
            )
        )
    # Calculate fdr from self_between and score
    if do_fdr and input_cols['fdr'] not in df.columns:
        fdr_ser = fdr.single_grouped_fdr(
            df.with_columns(
                score=pl.col(input_cols['score']),
                decoy_class=pl.col(input_cols['decoy_class']).replace(
                    [consts['tt_class'], consts['td_class'], consts['dt_class'], consts['dd_class']],
                    ['TT', 'TD', 'DT', 'DD']
                ),
                fdr_group=pl.col(input_cols['self_between']),
            ).with_columns(
                TT=pl.col('decoy_class')=='TT',
                TD=pl.col('decoy_class')=='TD',
                DD=pl.col('decoy_class')=='DD',
            ),
        )
        df = df.with_columns(
            fdr=fdr_ser
        )
    return df
