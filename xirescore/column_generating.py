import polars as pl

from xirescore.bi_fdr import self_or_between_mp, calculate_bi_fdr


def generate(df: pl.DataFrame, options: dict, do_self_between=False, do_fdr=False) -> pl.DataFrame:
    input_cols = options['input']['columns']
    cols_spectra = input_cols['spectrum_id']
    col_score = input_cols['score']
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
    # Generate decoy_class column from decoy_p1 and decoy_p2
    if input_cols['decoy_class'] not in df.columns:
        tt_expr = pl.col(input_cols['decoy_p1']).not_() & pl.col(input_cols['decoy_p2']).not_()
        dd_expr = pl.col(input_cols['decoy_p1']) & pl.col(input_cols['decoy_p2'])
        df = df.with_columns(
            decoy_class=pl.when(tt_expr).then(pl.lit('TT'))\
                          .when(dd_expr).then(pl.lit('DD'))\
                          .otherwise(pl.lit('TD'))
        )
    # Generate target column from decoy_class
    if input_cols['target'] not in df.columns:
        df.with_columns(
            (pl.col(input_cols['decoy_class']) == 'TT').alias(input_cols['target'])
        )
    # Calculte self_between from protein_p1, and protein_p2
    if do_self_between and input_cols['self_between'] not in df.columns:
        protein_p1_list = pl.col(input_cols['protein_p1'])
        protein_p2_list = pl.col(input_cols['protein_p2'])
        if df[input_cols['protein_p1']].dtype is pl.String:
            protein_p1_list = protein_p1_list.str.split(';')
        if df[input_cols['protein_p2']].dtype is pl.String:
            protein_p2_list = protein_p2_list.str.split(';')
        protein_p1_list = protein_p1_list.list.eval(
            pl.element().str.replace_all(options['input']['constants']['decoy_adjunct'], '')
        )
        protein_p2_list = protein_p2_list.list.eval(
            pl.element().str.replace_all(options['input']['constants']['decoy_adjunct'], '')
        )
        overlap_expr = protein_p1_list.list.set_intersection(protein_p2_list)
        df = df.with_columns(
            pl.when(overlap_expr == 0).then(pl.lit('between')).otherwise(pl.lit('self'))
        )
    # Calculate fdr from self_between and score
    if do_fdr and input_cols['fdr'] not in df.columns:
        fdr_pd = calculate_bi_fdr(
            df.filter(input_cols['top_ranking']).to_pandas(),
            score_col=input_cols['score'],
            decoy_class=input_cols['decoy_class'],
            fdr_group_col=input_cols['self_between'],
        )
        df = df.with_columns(
            fdr=pl.Series(fdr_pd)
        )
    return df
