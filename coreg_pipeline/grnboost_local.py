#!/usr/bin/env python
"""
GRNBoost2 without dask, for sandboxed environments.

WHY THIS EXISTS. arboreto's public entry points (`arboreto.algo.grnboost2`,
`diy`) construct a dask LocalCluster, which cannot start here:

    RuntimeError: Cluster failed to start: [Errno 1] Operation not permitted

pySCENIC ships `arboreto_with_multiprocessing.py` for exactly this situation.
This module is the same pattern.

WHY THE RESULTS ARE UNCHANGED. The unit of work is
`arboreto.core.infer_partial_network` -- precisely the per-target regression
that `create_graph` schedules onto dask workers. Regressor type, regressor
kwargs (`SGBM_KWARGS`: learning_rate 0.01, n_estimators 5000, max_features 0.1,
subsample 0.9) and the early-stopping window (25) are taken from arboreto
itself rather than restated, so importances match a dask run. Only the
scheduler differs; the documented trade-off is losing multi-node execution.

TARGET SUBSETTING IS EXACT. `create_graph` builds one independent regression
per target, so computing a subset of targets returns exactly the importances a
full run would give for those targets -- it is not an approximation.

Adapted from `run_cohort_grn.py` in the parent project, which established this
workaround; kept here so the pipeline directory is self-contained.
"""
from __future__ import annotations

import multiprocessing as mpr

import numpy as np
import pandas as pd


def _infer_one(args):
    """One target's regression. Module-level so multiprocessing can pickle it."""
    from arboreto.core import infer_partial_network
    (tf_matrix, tf_names, target_name, target_expr, kwargs, esw, seed) = args
    try:
        return infer_partial_network(
            regressor_type="GBM", regressor_kwargs=kwargs,
            tf_matrix=tf_matrix, tf_matrix_gene_names=tf_names,
            target_gene_name=target_name, target_gene_expression=target_expr,
            include_meta=False, early_stop_window_length=esw, seed=seed)
    except Exception:
        return None


def grnboost2_mp(expression_data, gene_names, tf_names, target_genes,
                 n_estimators=None, seed=0, n_workers=None, verbose=True,
                 progress_every=2000):
    """
    GRNBoost2 over a target subset via multiprocessing.

    expression_data : samples x genes (DataFrame or array)
    gene_names      : column names of expression_data
    tf_names        : regulators to use
    target_genes    : targets to fit
    Returns a DataFrame with columns TF, target, importance.
    """
    from arboreto.core import (SGBM_KWARGS, EARLY_STOP_WINDOW_LENGTH,
                               to_tf_matrix)

    kwargs = dict(SGBM_KWARGS)
    if n_estimators:
        kwargs["n_estimators"] = n_estimators
    n_workers = n_workers or max(1, mpr.cpu_count() - 2)

    matrix = (expression_data.values
              if hasattr(expression_data, "values") else np.asarray(expression_data))
    gene_names = list(gene_names)
    tf_matrix, tf_matrix_names = to_tf_matrix(matrix, gene_names, list(tf_names))

    idx = {g: i for i, g in enumerate(gene_names)}
    jobs = [(tf_matrix, tf_matrix_names, g, matrix[:, idx[g]],
             kwargs, EARLY_STOP_WINDOW_LENGTH, seed)
            for g in target_genes if g in idx]
    if not jobs:
        raise RuntimeError("no requested target is present in gene_names")
    if verbose:
        print(f"    GRNBoost2 (multiprocessing): {len(jobs):,} targets, "
              f"{n_workers} workers, {tf_matrix.shape[1]:,} regulators, "
              f"seed {seed}", flush=True)

    out, done, failed = [], 0, 0
    with mpr.Pool(n_workers) as pool:
        for res in pool.imap_unordered(_infer_one, jobs, chunksize=1):
            done += 1
            if res is not None and len(res):
                out.append(res)
            else:
                failed += 1
            if verbose and done % progress_every == 0:
                print(f"      {done:,}/{len(jobs):,} targets "
                      f"({len(out):,} produced edges)", flush=True)
    if not out:
        raise RuntimeError("no target produced a network")
    if verbose and failed:
        print(f"    {failed:,} of {len(jobs):,} targets produced no edges")
    return (pd.concat(out, ignore_index=True)
              .sort_values("importance", ascending=False)
              .reset_index(drop=True))
