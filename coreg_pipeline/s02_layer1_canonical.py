#!/usr/bin/env python
"""
Stage 02 -- Layer 1: miRNA -> gene (and miRNA -> TF) edges.

Two steps, deliberately separated:

  A. CANDIDATES. Read the curated multi-database merge and repair `n_sources`.
     The supplied table has n_sources == 1 on every one of its 6.57M rows
     because the builder split the source field on ";" while the field is
     comma-delimited. Multi-database agreement -- the natural evidence-strength
     axis in a 4-database merge -- was therefore never available. We re-derive
     it here by splitting on both separators.

  B. COHORT-SPECIFIC FILTER. Follow SPONGE's first step: for each gene, regress
     its expression on its candidate miRNAs with elastic net constrained to
     NON-POSITIVE coefficients, and keep the miRNAs retained with a negative
     coefficient. The constraint encodes the biology (miRNAs repress) and is
     what makes the filter more than a correlation screen.

CIRCULARITY -- the reason for two modes.

  A cohort-specific L1 selected on the same expression matrix that a downstream
  test then uses is not neutral, and the direction differs by layer. On the TF
  side, selecting edges on cohort expression inflates the variance the TF block
  absorbs, which makes a "miRNA explains variance beyond TFs" test CONSERVATIVE
  -- an acceptable bias. On the miRNA side it is the opposite: selecting miRNA
  edges by requiring negative miRNA-gene association and then testing whether
  those miRNAs explain target variance makes the selection criterion and the
  test statistic THE SAME QUANTITY. That is anti-conservative and the resulting
  p-values mean nothing.

  So:
    --mode full      selects on all samples. For NETWORK CONSTRUCTION only
                     (edges, feedback pairs, topology), where no significance
                     test is applied to the selected edges.
    --mode crossfit  splits samples into folds, selects on the out-of-fold
                     samples and emits per-fold edge sets with the held-in
                     sample ids, so a downstream test evaluates edges on data
                     that played no part in choosing them.

  Any combined-effect style test MUST consume the crossfit output. Feeding it
  the `full` output produces circular significance.

ALPHA SELECTION. `--alpha-rule cv` (default) mirrors cv.glmnet: pick alpha by
cross-validated error. `sparsest` (largest alpha keeping >=1 edge) is far more
aggressive -- it drops median retained miRNAs per gene from 8 to 5, pushing
genes below the >=5-regulator floor a downstream nested test needs -- and is
kept only as a sensitivity option.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as prov  # noqa: E402

from sklearn.linear_model import ElasticNet  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

BASE = Path(__file__).resolve().parent
EXPR = BASE / "data" / "expression"
REF = BASE / "data" / "reference"
OUT = BASE / "results_canonical" / "layer1"
OUT.mkdir(parents=True, exist_ok=True)

L1_RATIO = 0.5
N_ALPHAS = 12
ALPHA_MIN_RATIO = 1e-3
CV_FOLDS = 3        # cv.glmnet uses 10; 3 keeps folds x alphas x genes x
                    # networks tractable, and alpha choice is a coarse decision
MIN_CANDIDATES = 3  # genes with fewer candidate miRNAs are not fitted
MIN_EXPR_FRAC = 0.2
N_PROC = max(1, (os.cpu_count() or 4) - 2)


# ── A. candidates ────────────────────────────────────────────────────────────
def load_candidates(path: Path) -> pd.DataFrame:
    """Read the merge and recompute n_sources correctly."""
    c = pd.read_csv(path, sep="\t",
                    usecols=["target_symbol", "mirna", "sources", "target_is_tf"])
    c = c.dropna(subset=["target_symbol", "mirna"])

    # Repair: the field is comma-delimited; the original split on ";" only.
    src = c["sources"].fillna("").astype(str)
    c["n_sources"] = [len({t for t in s.replace(";", ",").split(",") if t})
                      for s in src]

    prov.record_input("mirna_gene_candidates", path=path, origin="local-supplied",
                      rows=len(c),
                      note="4-database merge (lncBase, microT_high, tarBase, "
                           "targetScan) supplied by user; n_sources recomputed "
                           "here because the shipped column was always 1")
    return c


# ── B. filter ────────────────────────────────────────────────────────────────
def fit_one_gene(y, X, alpha_rule="cv"):
    """
    Elastic net with coefficients constrained non-positive.

    sklearn has no upper_limits, so the constraint is imposed exactly by
    negating the predictors and requiring non-negative coefficients: fitting
    y ~ (-X) with positive=True and returning -coef gives precisely the
    feasible set glmnet's upper.limits=0 defines.
    """
    n, p = X.shape
    if n < 3 * CV_FOLDS or p < 1:
        return None
    ys = y - y.mean()
    sd = ys.std()
    if sd == 0:
        return None
    yz = ys / sd

    amax = float(np.abs(X.T @ yz).max() / max(n, 1))
    if not np.isfinite(amax) or amax <= 0:
        return None
    alphas = np.logspace(np.log10(amax), np.log10(amax * ALPHA_MIN_RATIO), N_ALPHAS)

    def _fit(a, Xf, yf):
        m = ElasticNet(alpha=a, l1_ratio=L1_RATIO, positive=True,
                       fit_intercept=True, max_iter=5000, tol=1e-3,
                       selection="cyclic")
        m.fit(Xf, yf)
        return m

    if alpha_rule == "cv":
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
        err = np.full((CV_FOLDS, len(alphas)), np.nan)
        for k, (tr, te) in enumerate(kf.split(X)):
            for j, a in enumerate(alphas):
                try:
                    m = _fit(a, X[tr], yz[tr])
                    err[k, j] = float(np.mean((yz[te] - m.predict(X[te])) ** 2))
                except Exception:
                    pass
        mean_err = np.nanmean(err, axis=0)
        if np.all(np.isnan(mean_err)):
            return None
        best = int(np.nanargmin(mean_err))
        try:
            m = _fit(alphas[best], X, yz)
        except Exception:
            return None
        return -m.coef_ if np.any(m.coef_ > 0) else None

    # sparsest: largest alpha that still retains an edge
    for a in alphas:
        try:
            m = _fit(a, X, yz)
        except Exception:
            continue
        if np.any(m.coef_ > 0):
            return -m.coef_
    return None


def _job(args):
    sym, mirs, y, Xm, rule = args
    coef = fit_one_gene(y, -Xm, alpha_rule=rule)
    if coef is None:
        return []
    return [(sym, m, float(c)) for m, c in zip(mirs, coef) if c < 0]


def build_jobs(gene, mirna, cand, cols, rule):
    g, m = gene[cols], mirna[cols]
    m_index = set(m.index)
    by_gene = cand.groupby("target_symbol")["mirna"].apply(list)
    jobs = []
    for sym, mirs in by_gene.items():
        if sym not in g.index:
            continue
        mirs = [x for x in dict.fromkeys(mirs) if x in m_index]
        if len(mirs) < MIN_CANDIDATES:
            continue
        y = g.loc[sym].to_numpy(dtype=float)
        Xm = m.loc[mirs].to_numpy(dtype=float).T
        if not np.isfinite(y).all() or not np.isfinite(Xm).all():
            ok = np.isfinite(y) & np.isfinite(Xm).all(axis=1)
            if ok.sum() < 3 * CV_FOLDS:
                continue
            y, Xm = y[ok], Xm[ok]
        sd = Xm.std(axis=0)
        keep = sd > 0
        if keep.sum() < MIN_CANDIDATES:
            continue
        mirs = [x for x, k in zip(mirs, keep) if k]
        Xm = (Xm[:, keep] - Xm[:, keep].mean(axis=0)) / sd[keep]
        jobs.append((sym, mirs, y, Xm, rule))
    return jobs


def run_selection(gene, mirna, cand, cols, rule, label=""):
    t0 = time.time()
    jobs = build_jobs(gene, mirna, cand, cols, rule)
    print(f"    {label}{len(jobs):,} genes to fit on {N_PROC} procs", flush=True)
    rows = []
    with Pool(processes=N_PROC) as pool:
        for i, r in enumerate(pool.imap_unordered(_job, jobs, chunksize=64), 1):
            rows.extend(r)
            if i % 4000 == 0:
                print(f"    {label}{i:,}/{len(jobs):,} genes, {len(rows):,} edges "
                      f"({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows, columns=["target_symbol", "mirna", "coef"]), len(jobs)


# ── main ─────────────────────────────────────────────────────────────────────
def expressed(gene, mirna):
    n = gene.shape[1]
    gk = gene.index[(gene > 0).sum(axis=1) >= MIN_EXPR_FRAC * n]
    mk = mirna.index[(mirna > 0).sum(axis=1) >= MIN_EXPR_FRAC * n]
    return set(gk), set(mk)


def process(network: str, cand: pd.DataFrame, mode: str, rule: str,
            n_folds: int, seed: int):
    d = EXPR / network
    gene = pd.read_parquet(d / "gene_log2.parquet")
    mirna = pd.read_parquet(d / "mirna_log2.parquet")
    meta = json.loads((d / "meta.json").read_text())
    cols = list(gene.columns)
    print(f"\n=== {network} ({len(cols)} samples, {meta['sample_type']}) ===")

    gk, mk = expressed(gene, mirna)
    c = cand[cand["target_symbol"].isin(gk) & cand["mirna"].isin(mk)]
    print(f"  expressed: {len(gk):,} genes, {len(mk):,} miRNAs "
          f"-> {len(c):,} candidate edges")

    if mode == "full":
        edges, n_fit = run_selection(gene, mirna, c, cols, rule)
        out = OUT / f"l1_{network}_full.tsv"
        edges.to_csv(out, sep="\t", index=False)
        per_gene = edges.groupby("target_symbol")["mirna"].nunique()
        print(f"  kept {len(edges):,}/{len(c):,} ({len(edges)/max(len(c),1)*100:.2f}%)"
              f" | median miRNAs/gene {per_gene.median():.0f}")
        (OUT / f"l1_{network}_full_meta.json").write_text(json.dumps({
            "network": network, "mode": "full", "alpha_rule": rule,
            "cv_folds": CV_FOLDS if rule == "cv" else None,
            "n_samples": len(cols), "n_genes_fitted": n_fit,
            "n_candidates": int(len(c)), "n_edges": int(len(edges)),
            "USE_FOR": "network construction only -- selected on all samples, "
                       "so any significance test on these edges is circular",
        }, indent=2))
        prov.record_output(f"l1_{network}_full", path=out, rows=len(edges))
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        allf = []
        for k, (sel_i, test_i) in enumerate(kf.split(cols)):
            sel = [cols[i] for i in sel_i]
            test = [cols[i] for i in test_i]
            e, n_fit = run_selection(gene, mirna, c, sel, rule, label=f"fold{k} ")
            e["fold"] = k
            allf.append(e)
            pd.DataFrame({"barcode": test}).to_csv(
                OUT / f"l1_{network}_crossfit_fold{k}_testsamples.tsv",
                sep="\t", index=False)
            print(f"  fold {k}: selected on {len(sel)}, held out {len(test)}, "
                  f"{len(e):,} edges")
        edges = pd.concat(allf, ignore_index=True)
        out = OUT / f"l1_{network}_crossfit.tsv"
        edges.to_csv(out, sep="\t", index=False)
        (OUT / f"l1_{network}_crossfit_meta.json").write_text(json.dumps({
            "network": network, "mode": "crossfit", "alpha_rule": rule,
            "n_folds": n_folds, "seed": seed, "n_samples": len(cols),
            "n_edges": int(len(edges)),
            "USE_FOR": "significance testing -- evaluate fold k's edges ONLY on "
                       "that fold's held-out barcodes",
        }, indent=2))
        prov.record_output(f"l1_{network}_crossfit", path=out, rows=len(edges))


def main(networks=None, candidates=None, mode="full", rule="cv",
         n_folds=2, seed=0):
    cpath = Path(candidates or
                 BASE.parent / "coregulatory_network" / "mirna_gene_interactions.tsv")
    if not cpath.exists():
        raise SystemExit(f"missing candidate table: {cpath}")
    print("=== stage 02: Layer 1 ===")
    cand = load_candidates(cpath)
    ns = cand["n_sources"].value_counts().sort_index()
    print(f"  candidates {len(cand):,} | n_sources {ns.to_dict()}")
    print(f"  miRNAs {cand['mirna'].nunique():,} | targets "
          f"{cand['target_symbol'].nunique():,}")

    avail = sorted(d.name for d in EXPR.iterdir()
                   if d.is_dir() and (d / "meta.json").exists())
    chosen = networks or avail
    missing = [n for n in chosen if n not in avail]
    if missing:
        raise SystemExit(f"no expression matrix for: {missing}")
    print(f"  networks: {len(chosen)}")

    for net in chosen:
        process(net, cand, mode, rule, n_folds, seed)
    print("\ndone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="*", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--mode", choices=["full", "crossfit"], default="full")
    ap.add_argument("--alpha-rule", choices=["cv", "sparsest"], default="cv")
    ap.add_argument("--n-folds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(networks=a.networks, candidates=a.candidates, mode=a.mode,
         rule=a.alpha_rule, n_folds=a.n_folds, seed=a.seed)
