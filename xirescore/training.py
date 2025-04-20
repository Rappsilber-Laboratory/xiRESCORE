from typing import Callable
import importlib
import logging

import polars as pl
from sklearn.base import ClassifierMixin
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import multiprocess as mp

from xirescore import async_result_resolving
from xirescore.NoOverlapKFold import NoOverlapKFold


logger = logging.getLogger(__name__)

def train(train_df: pl.DataFrame, cols_features, clf_params, options, splits=None):
    # Get peptide sequence columns
    cols_pepseq = [
        options['input']['columns']['base_sequence_p1'],
        options['input']['columns']['base_sequence_p2'],
    ]

    # Get column for target labeling
    col_label = options['input']['columns']['is_tt']

    # Get DataFrames for peptide sequences, features and labels
    pepseq_df = train_df.select(cols_pepseq)
    features_df = train_df.select(cols_features)
    labels_df = train_df[col_label]

    # Import model
    model_class = options['rescoring']['model_class']
    model_name = options['rescoring']['model_name']
    model_module = importlib.import_module(f"sklearn.{model_class}")
    model: ClassifierMixin.__class__ = getattr(model_module, model_name)

    # Check for single core model
    is_mp_model = hasattr(model, 'n_jobs')

    # Get random seed and number of k-fold splits
    seed = options['rescoring']['random_seed']
    n_splits = options['rescoring']['n_splits']

    if splits is None:
        # Get non peptide overlapping k-fold splits
        kf = NoOverlapKFold(
            n_splits,
            pep1_id_col=options['input']['columns']['base_sequence_p1'],
            pep2_id_col=options['input']['columns']['base_sequence_p2'],
            target_col=col_label,
            random_state=seed,
            shuffle=True,
        )
        splits = kf.splits_by_peptides(
            df=train_df.to_pandas(),
            pepseqs=pepseq_df.to_pandas()
        )

    max_jobs = options['rescoring']['max_jobs']
    if max_jobs < 1:
        max_jobs = mp.cpu_count() - 1

    fold_clfs = []
    with mp.get_context("spawn").Pool(processes=max_jobs) as pool:
        for cvi, fold in enumerate(splits):
            # Only run multiprocessing for single core models
            if is_mp_model:
                fold_clf = [
                    lambda: train_fold(
                        features_df=features_df,
                        labels_df=labels_df,
                        fold=fold,
                        params=clf_params,
                        options=options
                    )
                ]
            else:
                fold_clf = pool.apply_async(
                    train_fold,
                    kwds=dict(
                        features_df=features_df,
                        labels_df=labels_df,
                        fold=fold,
                        params=clf_params,
                        options=options
                    )
                )
            # Add job to the
            fold_clfs.append(fold_clf)
        # Resolve (potentially) async results
        clfs = async_result_resolving.resolve(fold_clfs)
    return clfs, splits


def train_fold(features_df, labels_df, fold, params, options):
    # Import classifier model
    model_class = options['rescoring']['model_class']
    model_name = options['rescoring']['model_name']
    model_module = importlib.import_module(f"sklearn.{model_class}")
    model: ClassifierMixin.__class__ = getattr(model_module, model_name)

    # Import metric function
    metric_name = options['rescoring']['metric_name']
    metric: Callable = getattr(sklearn.metrics, metric_name)

    # Unpack fold
    train_idx, test_idx = fold

    # Get fold's train features and labels
    fold_train_features_df = features_df[train_idx.to_numpy()]
    fold_train_labels_df = labels_df[train_idx.to_numpy()]

    # Get fold's test features and labels
    fold_test_features_df = features_df[test_idx.to_numpy()]
    fold_test_labels_df = labels_df[test_idx.to_numpy()]

    # Train fold classifier
    clf = model(**params)
    clf.fit(
        fold_train_features_df,
        fold_train_labels_df
    )
    test_predictions = clf.predict(fold_test_features_df)

    # Evaluate fold model
    score = metric(
        fold_test_labels_df.to_numpy(),
        test_predictions,
        labels=[0, 1]
    )
    accuracy = accuracy_score(
        fold_test_labels_df.to_numpy(),
        test_predictions,
    )
    balanced_accuracy = balanced_accuracy_score(
        fold_test_labels_df.to_numpy(),
        test_predictions,
    )
    logger.info(f"Score: {score}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info(f"Balanced accuracy: {balanced_accuracy*100:.2f}%")

    return clf
