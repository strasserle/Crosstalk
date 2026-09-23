#!/usr/bin/env python
"""
Stage 05 -- what miRNA regulation actually contributes, three ways.

BACKGROUND. The predecessor analysis ("combined effect") fitted target ~ TFs
versus target ~ TFs + miRNAs and F-tested the miRNA block. That statistic is the
semi-partial R^2 of the miRNA block given TFs -- the miRNA-UNIQUE contribution,
not a combined effect: the shared TF/miRNA variance is exactly what it partials
out. Worse, the F test never inspects coefficient signs, so it fires identically
on a negative miRNA-target association (repression, as the mechanism predicts)
and a positive one (which contradicts it). Empirically only 46.1% of its
FDR-significant pairs across 23 cohorts pointed negative -- sign-random, tilted
slightly the wrong way. So the headline was largely reporting bulk co-expression
structure, not repression.

This stage replaces it with three quantities that each mean something.

  (1) SIGN-CONSTRAINED EFFECT. Partial correlation of each (miRNA, target) pair
      controlling for the target's TF block, tested ONE-SIDED for negative
      association. Mechanism-aligned: a positive association is now evidence
      against, not for.

  (2) VARIANCE PARTITION (commonality analysis). Per target, R^2 of the TF-only,
      miRNA-only and joint models, decomposed into unique-TF, unique-miRNA and
      SHARED. The shared term is what "coregulation" should mean -- two
      regulator classes explaining the same variance -- and it is precisely
      what the old semi-partial statistic discarded. Note shared can be
      NEGATIVE (suppression / enhancement), which is a real and interpretable
      configuration, not an error.

  (3) MEDIATION ON FEED-FORWARD TRIADS. For TF -> miRNA -> target with a direct
      TF -> target edge, decompose the TF's effect into a direct path (c') and
      an indirect path through the miRNA (a*b). This is the analysis the
      coregulatory topology uniquely enables, it is signed, and it makes
      coherent vs incoherent FFLs empirically testable rather than merely
      annotated.

      a  : TF -> miRNA          (miRNA ~ TF)
      b  : miRNA -> target      (target ~ TF + miRNA, miRNA coefficient)
      c' : TF -> target direct  (same fit, TF coefficient)
      indirect = a*b ; total = c' + a*b
      Coherent iff sign(c') == sign(a*b).

      Significance by the delta-method (Sobel) SE on a*b. Sobel is known
      conservative for small samples and non-normal products; it is used here
      because it is closed-form over ~77k triads, and the FDR-passing subset can
      be bootstrapped separately if a specific triad carries a claim.

DF HANDLING. Degrees of freedom come from the numerical RANK of each design,
not its column count. Regulator blocks are strongly collinear, so column counts
overstate consumed df and inflate F -- an error that produced a spuriously large
significant fraction earlier in this project.

INPUTS are stage 01 expression plus stage 02/03 edges filtered by stage 04's
reproducibility threshold. Expression is already log2; it is standardised here,
never re-transformed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as prov  # noqa: E402

BASE = Path(__file__).resolve().parent
EXPR = BASE / "data" / "expression"
REF = BASE / "data" / "reference"
L1D = BASE / "results" / "layer1"
L2D = BASE / "results" / "layer2"
NETD = BASE / "results" / "network"
OUT = BASE / "results" / "effects"
OUT.mkdir(parents=True, exist_ok=True)

MIN_TF = 3          # a target needs at least this many TF regulators
MIN_MIR = 3         # ... and this many miRNA regulators
MIN_RESID_DF = 20   # ... leaving at least this much residual df
MAX_TF = 40         # cap block width so residual df survives (ranked by importance)
MAX_MIR = 40


def rank_of(X: np.ndarray) -> int:
    return int(np.linalg.matrix_rank(X)) if X.size else 0


def design(X: np.ndarray) -> np.ndarray:
    """Add an intercept column."""
    return np.column_stack([np.ones(len(X)), X]) if X.size else np.ones((0, 1))


def rss_r2(y: np.ndarray, X: np.ndarray) -> tuple[float, float, int]:
    """Residual sum of squares, R^2 and consumed rank for OLS of y on X."""
    Xd = design(X)
    coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ coef
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    return rss, r2, rank_of(Xd)


def zscore(A: np.ndarray) -> np.ndarray:
    sd = A.std(axis=0)
    sd[sd == 0] = 1.0
    return (A - A.mean(axis=0)) / sd


# ── (1) + (2) per-target ─────────────────────────────────────────────────────
def per_target(network: str, gene, mirna, tf_map, mir_map):
    rows, pair_rows = [], []
    t0 = time.time()
    targets = sorted(set(tf_map) & set(mir_map) & set(gene.index))
    print(f"  targets with both blocks: {len(targets):,}")

    for i, tgt in enumerate(targets, 1):
        tfs = [t for t in tf_map[tgt] if t in gene.index and t != tgt][:MAX_TF]
        mirs = [m for m in mir_map[tgt] if m in mirna.index][:MAX_MIR]
        if len(tfs) < MIN_TF or len(mirs) < MIN_MIR:
            continue
        y = gene.loc[tgt].to_numpy(float)
        Z = gene.loc[tfs].to_numpy(float).T          # TF block
        M = mirna.loc[mirs].to_numpy(float).T        # miRNA block
        ok = np.isfinite(y) & np.isfinite(Z).all(axis=1) & np.isfinite(M).all(axis=1)
        if ok.sum() < MIN_RESID_DF + len(tfs) + len(mirs):
            continue
        y, Z, M = y[ok], zscore(Z[ok]), zscore(M[ok])
        n = len(y)
        y = y - y.mean()
        if y.std() == 0:
            continue

        rss_tf, r2_tf, rk_tf = rss_r2(y, Z)
        rss_mir, r2_mir, rk_mir = rss_r2(y, M)
        rss_j, r2_j, rk_j = rss_r2(y, np.column_stack([Z, M]))
        df_res = n - rk_j
        if df_res < MIN_RESID_DF or rss_j <= 0:
            continue

        # (2) commonality decomposition.
        #
        # Computed on UNADJUSTED R^2 because the identity
        #   shared = R2_tf + R2_mir - R2_joint
        # only holds for unadjusted values. Unadjusted R^2 rises mechanically
        # with predictor count, so with blocks up to MAX_TF + MAX_MIR wide the
        # absolute levels are inflated; adjusted values are carried alongside so
        # the inflation is visible and the decomposition can be re-derived on
        # whichever basis a reader prefers.
        uniq_mir = r2_j - r2_tf
        uniq_tf = r2_j - r2_mir
        shared = r2_tf + r2_mir - r2_j

        def adj(r2, rk):
            return 1 - (1 - r2) * (n - 1) / max(n - rk, 1)
        r2_tf_a, r2_mir_a, r2_j_a = adj(r2_tf, rk_tf), adj(r2_mir, rk_mir), adj(r2_j, rk_j)

        # nested F on the miRNA block (the OLD statistic, kept for comparison)
        k_ex = rk_j - rk_tf
        if k_ex > 0:
            F = ((rss_tf - rss_j) / k_ex) / (rss_j / df_res)
            p_block = float(stats.f.sf(F, k_ex, df_res))
        else:
            F, p_block = np.nan, np.nan

        rows.append({"target": tgt, "n": n, "n_tf": len(tfs), "n_mir": len(mirs),
                     "r2_tf": r2_tf, "r2_mir": r2_mir, "r2_joint": r2_j,
                     "r2_tf_adj": r2_tf_a, "r2_mir_adj": r2_mir_a,
                     "r2_joint_adj": r2_j_a,
                     "unique_tf": uniq_tf, "unique_mir": uniq_mir,
                     "shared": shared, "resid_df": df_res,
                     "rank_tf": rk_tf, "rank_joint": rk_j,
                     "F_mir_block": F, "p_mir_block": p_block})

        # (1) sign-constrained per-pair partial correlation, miRNA vs target
        #     controlling for the TF block. One-sided for NEGATIVE.
        Zd = design(Z)
        cy, *_ = np.linalg.lstsq(Zd, y, rcond=None)
        ry = y - Zd @ cy
        cM, *_ = np.linalg.lstsq(Zd, M, rcond=None)
        rM = M - Zd @ cM
        dfp = n - rk_tf - 1
        if dfp < MIN_RESID_DF:
            continue
        sy = ry.std()
        for j, m in enumerate(mirs):
            sm = rM[:, j].std()
            if sy == 0 or sm == 0:
                continue
            r = float(np.dot(ry, rM[:, j]) / (len(ry) * sy * sm))
            r = max(min(r, 0.999999), -0.999999)
            t = r * np.sqrt(dfp / (1 - r * r))
            pair_rows.append({"target": tgt, "mirna": m, "r_partial": r,
                              "df": dfp,
                              "p_two_sided": float(2 * stats.t.sf(abs(t), dfp)),
                              "p_one_sided_neg": float(stats.t.cdf(t, dfp))})
        if i % 2000 == 0:
            print(f"    {i:,}/{len(targets):,} targets ({time.time()-t0:.0f}s)",
                  flush=True)

    return pd.DataFrame(rows), pd.DataFrame(pair_rows)


# ── (3) mediation over triads ────────────────────────────────────────────────
def mediation(triads: pd.DataFrame, gene, mirna):
    """
    Vectorised where possible: `a` depends only on (TF, miRNA), so it is fitted
    once per pair; `b` and `c'` need the per-triad two-predictor fit.
    """
    pairs = triads[["TF", "miRNA"]].drop_duplicates()
    a_map = {}
    for tf, mir in pairs.itertuples(index=False):
        if tf not in gene.index or mir not in mirna.index:
            continue
        x = gene.loc[tf].to_numpy(float)
        m = mirna.loc[mir].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(m)
        if ok.sum() < MIN_RESID_DF:
            continue
        x, m = x[ok], m[ok]
        if x.std() == 0 or m.std() == 0:
            continue
        xz = (x - x.mean()) / x.std()
        mz = (m - m.mean()) / m.std()
        n = len(xz)
        a = float(np.dot(xz, mz) / n)                       # standardised slope
        resid = mz - a * xz
        se_a = float(np.sqrt((resid @ resid) / (n - 2) / n))
        a_map[(tf, mir)] = (a, se_a, n)
    print(f"  TF->miRNA paths fitted: {len(a_map):,}/{len(pairs):,}")

    rows = []
    t0 = time.time()
    for i, (tf, mir, tgt) in enumerate(
            triads[["TF", "miRNA", "target"]].itertuples(index=False), 1):
        key = (tf, mir)
        if key not in a_map or tgt not in gene.index:
            continue
        a, se_a, _ = a_map[key]
        x = gene.loc[tf].to_numpy(float)
        m = mirna.loc[mir].to_numpy(float)
        y = gene.loc[tgt].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(m) & np.isfinite(y)
        if ok.sum() < MIN_RESID_DF:
            continue
        x, m, y = x[ok], m[ok], y[ok]
        if x.std() == 0 or m.std() == 0 or y.std() == 0:
            continue
        X = np.column_stack([(x - x.mean()) / x.std(), (m - m.mean()) / m.std()])
        yz = (y - y.mean()) / y.std()
        Xd = design(X)
        coef, *_ = np.linalg.lstsq(Xd, yz, rcond=None)
        resid = yz - Xd @ coef
        n = len(yz)
        dfr = n - rank_of(Xd)
        if dfr < MIN_RESID_DF:
            continue
        sigma2 = float(resid @ resid) / dfr
        try:
            cov = sigma2 * np.linalg.pinv(Xd.T @ Xd)
        except np.linalg.LinAlgError:
            continue
        c_prime, b = float(coef[1]), float(coef[2])
        se_b = float(np.sqrt(max(cov[2, 2], 0)))
        indirect = a * b
        # Sobel / delta-method SE of the product
        se_ind = float(np.sqrt(max(a * a * se_b ** 2 + b * b * se_a ** 2, 0)))
        z = indirect / se_ind if se_ind > 0 else np.nan
        rows.append({"TF": tf, "miRNA": mir, "target": tgt, "n": n,
                     "a_tf_to_mir": a, "b_mir_to_tgt": b,
                     "c_prime_direct": c_prime, "indirect": indirect,
                     "total": c_prime + indirect, "se_indirect": se_ind,
                     "z_sobel": z,
                     "p_sobel": float(2 * stats.norm.sf(abs(z)))
                     if np.isfinite(z) else np.nan,
                     "coherent": bool(np.sign(c_prime) == np.sign(indirect))})
        if i % 20000 == 0:
            print(f"    {i:,}/{len(triads):,} triads ({time.time()-t0:.0f}s)",
                  flush=True)
    return pd.DataFrame(rows)


def load_l1_folds(network: str):
    """
    Cross-fitted L1: list of (fold, edges, held_out_barcodes).

    Fold k's edges were selected WITHOUT the barcodes returned here, so a sign
    or mediation test run on those barcodes is not evaluating its own selection
    criterion. This is the only L1 that may enter a significance test -- the
    `full` output selects on every sample, and a negativity test on it recovers
    the elastic-net non-positivity constraint rather than measuring anything.
    """
    p = L1D / f"l1_{network}_crossfit.tsv"
    if not p.exists():
        raise SystemExit(
            f"missing {p.name}. Run:\n"
            f"  python s02_layer1.py --networks {network} --mode crossfit")
    ed = pd.read_csv(p, sep="\t")
    out = []
    for k in sorted(ed["fold"].unique()):
        tp = L1D / f"l1_{network}_crossfit_fold{k}_testsamples.tsv"
        if not tp.exists():
            raise SystemExit(f"missing held-out sample list {tp.name}")
        held = pd.read_csv(tp, sep="\t")["barcode"].astype(str).tolist()
        out.append((int(k), ed[ed["fold"] == k], held))
    return out


def main(network="healthy_pooled", max_triads=None, crossfit=False):
    gene = pd.read_parquet(EXPR / network / "gene_log2.parquet")
    mirna = pd.read_parquet(EXPR / network / "mirna_log2.parquet")
    meta = json.loads((EXPR / network / "meta.json").read_text())
    print(f"=== stage 05: effects | {network} "
          f"({meta['n_samples']} samples, {meta['sample_type']}) ===")

    l1 = pd.read_csv(L1D / f"l1_{network}_full.tsv", sep="\t")
    summ = json.loads((NETD / f"summary_{network}.json").read_text())
    thr = summ["l2_importance_threshold"]
    l2p = L2D / f"l2_{network}_2seed.tsv"
    if not l2p.exists():
        l2p = L2D / f"l2_{network}.tsv"
    l2 = pd.read_csv(l2p, sep="\t", usecols=["tf", "target", "importance"])
    l2 = l2[l2["importance"] >= thr]
    print(f"  L1 {len(l1):,} edges | L2 {len(l2):,} edges above q-threshold "
          f"{thr:.4f}")

    tf_map = (l2.sort_values("importance", ascending=False)
                .groupby("target")["tf"].apply(list).to_dict())

    if crossfit:
        # Each fold's edges are evaluated ONLY on that fold's held-out samples.
        folds = load_l1_folds(network)
        print(f"  CROSSFIT: {len(folds)} folds "
              f"({[len(h) for _, _, h in folds]} held-out samples each)")
        tparts, pparts = [], []
        for k, ed, held in folds:
            cols = [c for c in held if c in gene.columns and c in mirna.columns]
            mm = ed.groupby("target_symbol")["mirna"].apply(list).to_dict()
            print(f"  -- fold {k}: {len(ed):,} selected edges, "
                  f"evaluating on {len(cols)} held-out samples")
            t_k, p_k = per_target(network, gene[cols], mirna[cols], tf_map, mm)
            t_k["fold"], p_k["fold"] = k, k
            tparts.append(t_k)
            pparts.append(p_k)
        tgt_df = pd.concat(tparts, ignore_index=True)
        pair_df = pd.concat(pparts, ignore_index=True)
    else:
        mir_map = l1.groupby("target_symbol")["mirna"].apply(list).to_dict()
        tgt_df, pair_df = per_target(network, gene, mirna, tf_map, mir_map)
    print(f"  per-target rows {len(tgt_df):,} | pair rows {len(pair_df):,}")

    if len(pair_df):
        pair_df["adj_p_two_sided"] = multipletests(
            pair_df["p_two_sided"], method="fdr_bh")[1]
        pair_df["adj_p_one_sided_neg"] = multipletests(
            pair_df["p_one_sided_neg"], method="fdr_bh")[1]
    if len(tgt_df):
        ok = tgt_df["p_mir_block"].notna()
        tgt_df["adj_p_mir_block"] = np.nan
        tgt_df.loc[ok, "adj_p_mir_block"] = multipletests(
            tgt_df.loc[ok, "p_mir_block"], method="fdr_bh")[1]

    triads = pd.read_csv(NETD / f"ffl_triads_{network}.tsv", sep="\t")
    if max_triads:
        triads = triads.head(max_triads)
    print(f"\n  mediation over {len(triads):,} FFL triads")
    med = mediation(triads, gene, mirna)
    if len(med):
        med["adj_p_sobel"] = multipletests(
            med["p_sobel"].fillna(1.0), method="fdr_bh")[1]

    sfx = "_crossfit" if crossfit else "_full"
    for nm, df in [("target_variance_partition", tgt_df),
                   ("pair_sign_constrained", pair_df),
                   ("ffl_mediation", med)]:
        p = OUT / f"{nm}_{network}{sfx}.tsv"
        df.to_csv(p, sep="\t", index=False)
        prov.record_output(f"{nm}_{network}{sfx}", path=p, rows=len(df))
        print(f"  wrote {len(df):,} rows -> {p.name}")

    # headline comparison, computed not asserted
    s = {}
    if len(tgt_df):
        s["n_targets"] = int(len(tgt_df))
        s["pct_old_block_F"] = float(
            (tgt_df["adj_p_mir_block"] < 0.05).mean() * 100)
        s["median_r2_joint"] = float(tgt_df["r2_joint"].median())
        s["median_unique_mir"] = float(tgt_df["unique_mir"].median())
        s["median_unique_tf"] = float(tgt_df["unique_tf"].median())
        s["median_shared"] = float(tgt_df["shared"].median())
        s["pct_shared_negative"] = float((tgt_df["shared"] < 0).mean() * 100)
    if len(pair_df):
        sig2 = pair_df["adj_p_two_sided"] < 0.05
        signeg = pair_df["adj_p_one_sided_neg"] < 0.05
        s["n_pairs"] = int(len(pair_df))
        s["n_pairs_sig_two_sided"] = int(sig2.sum())
        s["frac_negative_among_two_sided"] = float(
            (pair_df.loc[sig2, "r_partial"] < 0).mean()) if sig2.any() else None
        s["n_pairs_sig_negative_one_sided"] = int(signeg.sum())
        s["pct_targets_with_sig_negative_pair"] = float(
            pair_df.loc[signeg, "target"].nunique()
            / pair_df["target"].nunique() * 100)
    if len(med):
        msig = med["adj_p_sobel"] < 0.05
        s["n_triads"] = int(len(med))
        s["n_triads_sig_indirect"] = int(msig.sum())
        s["pct_coherent_all"] = float(med["coherent"].mean() * 100)
        s["pct_coherent_sig"] = float(
            med.loc[msig, "coherent"].mean() * 100) if msig.any() else None
        s["median_abs_indirect_sig"] = float(
            med.loc[msig, "indirect"].abs().median()) if msig.any() else None
        s["median_abs_direct_sig"] = float(
            med.loc[msig, "c_prime_direct"].abs().median()) if msig.any() else None
    s["mode"] = "crossfit" if crossfit else "full"
    s["sign_test_valid"] = bool(crossfit)
    # The --crossfit path covers (1) the sign test and (2) the variance
    # partition. It does NOT cover (3) mediation: triads come from stage 04,
    # which is built on the full-mode L1, and the `b` path is fitted on all
    # samples. So b is estimated on data that took part in selecting the
    # miRNA->target edge, biasing it toward significance and toward negative
    # sign. Mediation numbers are therefore IDENTICAL in both modes -- if they
    # differ, something is wrong. Treat the indirect effect as descriptive
    # until stage 04 gains a cross-fitted triad build.
    s["mediation_crossfitted"] = False
    s["mediation_note"] = (
        "Indirect-effect magnitudes and coherence fractions are DESCRIPTIVE. "
        "The b path (miRNA->target) is fitted on samples that participated in "
        "selecting that edge, so p_sobel is anti-conservative and the "
        "coherent fraction is biased upward. Direct path c' and the a path are "
        "unaffected. Needs a cross-fitted triad build to become inferential.")
    s["sign_test_note"] = (
        "Cross-fitted: each fold's miRNA edges were selected without the "
        "samples used to test them, so the negativity result is evidence."
        if crossfit else
        "NOT VALID as evidence of repression: L1 edges were selected for "
        "negative elastic-net coefficients on these same samples, so a "
        "negativity test recovers the selection constraint. Kept only for "
        "comparison against the cross-fitted run.")
    (OUT / f"summary_{network}{sfx}.json").write_text(json.dumps(s, indent=2))
    print("\nsummary:")
    for k, v in s.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="healthy_pooled")
    ap.add_argument("--max-triads", type=int, default=None,
                    help="cap triads (smoke tests only)")
    ap.add_argument("--crossfit", action="store_true",
                    help="use cross-fitted L1 and evaluate each fold only on "
                         "its held-out samples. REQUIRED for the sign test to "
                         "mean anything")
    a = ap.parse_args()
    main(network=a.network, max_triads=a.max_triads, crossfit=a.crossfit)
