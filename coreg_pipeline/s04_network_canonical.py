#!/usr/bin/env python
"""
Stage 04 -- assemble the coregulatory network from L1 and L2.

Produces, per network:

  tf_to_mirna.tsv   TF -> mature miRNA, bridged through the miRNA's host gene
  mirna_to_tf.tsv   mature miRNA -> TF (the L1 edges whose target is a TF)
  feedback.tsv      pairs present in BOTH directions
  ffl_triads.tsv    (TF, miRNA, target) where TF->miRNA, TF->target, miRNA->target

THE NAME BRIDGE IS THE RISKY PART. Three ID spaces meet here:

  L2 targets        HGNC gene symbols        ("MIR21", "MIR205HG", "TP53")
  loci table        Ensembl miRNA gene symbol("MIR21", "MIRLET7A1")
  L1 / expression   mature miRNA names       ("hsa-miR-21-5p", "hsa-let-7a-5p")

A miRNA gene yields one or two mature arms (-5p/-3p), and the naming conventions
differ in prefix, case, and hyphenation. `mirna_core()` strips both spaces to a
comparable core token, and the match rate is REPORTED and asserted non-trivial
rather than assumed -- a silent failure here yields an empty network that still
writes successfully.

EDGE FILTERING follows the reproducibility result measured in stage 03: two
GRNBoost2 seeds on the identical matrix agree at Jaccard 0.265 overall, but
0.761 in the top importance decile. So L2 edges are filtered to the top decile
by default (`--l2-quantile 0.9`); below that a single-seed edge is more likely
absent than present on a rerun and has no business in a triad.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as prov  # noqa: E402

BASE = Path(__file__).resolve().parent
REF = BASE / "data" / "reference"
L1D = BASE / "results_canonical" / "layer1"
L2D = BASE / "results" / "layer2"
OUT = BASE / "results_canonical" / "network"
OUT.mkdir(parents=True, exist_ok=True)


def mirna_core(name: str) -> str:
    """
    Reduce a miRNA identifier from any of the three spaces to a core token.

    "hsa-miR-21-5p" -> "mir21"      "MIR21"     -> "mir21"
    "hsa-let-7a-5p" -> "let7a"      "MIRLET7A1" -> "let7a1"
    "hsa-miR-199a-3p" -> "mir199a"

    Arm suffixes (-5p/-3p) and the species prefix are dropped, since a miRNA
    GENE maps to both arms; the caller decides how to handle the one-to-many
    expansion. Trailing copy-number digits on Ensembl symbols (MIRLET7A1) are
    NOT stripped, because they distinguish paralogous loci -- so those match a
    mature name only when the mature name carries the same suffix.
    """
    s = str(name).lower().replace("_", "-")
    s = re.sub(r"^(hsa|mmu|rno)-", "", s)
    s = re.sub(r"-(5p|3p)$", "", s)
    s = s.replace("-", "")
    s = re.sub(r"^micrornalet", "let", s)
    s = re.sub(r"^mirlet", "let", s)
    return s


def build_bridge(loci: pd.DataFrame, mature_names: list[str]) -> pd.DataFrame:
    """
    miRNA gene symbol (loci) -> mature miRNA name (L1/expression).

    Returns long format: one row per (mirna_gene, host_symbol, host_type,
    mature) combination, so a gene with two arms contributes two rows.
    """
    mat = pd.DataFrame({"mature": mature_names})
    mat["core"] = mat["mature"].map(mirna_core)

    lo = loci.dropna(subset=["symbol"]).copy()
    lo["core"] = lo["symbol"].map(mirna_core)

    # Exact core match first.
    br = lo.merge(mat, on="core", how="inner")

    # Ensembl copy-number suffixes (MIRLET7A1 -> let7a1) have no mature
    # counterpart when the mature name is unsuffixed (hsa-let-7a-5p -> let7a).
    # Retry the unmatched loci against a de-suffixed core.
    unmatched = lo[~lo["core"].isin(set(br["core"]))].copy()
    if len(unmatched):
        unmatched["core2"] = unmatched["core"].str.replace(r"\d+$", "", regex=True)
        br2 = unmatched.merge(mat.rename(columns={"core": "core2"}),
                              on="core2", how="inner").drop(columns="core2")
        br = pd.concat([br, br2], ignore_index=True)

    br = br.drop_duplicates(subset=["symbol", "host_symbol", "mature"])
    return br


def assemble(network: str, l2_quantile: float, min_l2_edges: int):
    l1p = L1D / f"l1_{network}_full.tsv"
    l2p = L2D / f"l2_{network}_2seed.tsv"
    if not l2p.exists():
        l2p = L2D / f"l2_{network}.tsv"
    for p in (l1p, l2p):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    print(f"\n=== {network} ===")
    l1 = pd.read_csv(l1p, sep="\t")                 # target_symbol, mirna, coef
    l2 = pd.read_csv(l2p, sep="\t")                 # tf, target, importance, ...
    loci = pd.read_csv(REF / "mirna_loci.tsv", sep="\t")
    print(f"  L1 {len(l1):,} edges | L2 {len(l2):,} edges | loci {len(loci):,} rows")

    # ── L2 filtering on the reproducibility threshold ────────────────────────
    thr = l2["importance"].quantile(l2_quantile)
    l2f = l2[l2["importance"] >= thr].copy()
    if len(l2f) < min_l2_edges:
        raise SystemExit(f"{network}: only {len(l2f):,} L2 edges above q"
                         f"{l2_quantile}; refusing to build on that")
    print(f"  L2 filtered to top {(1-l2_quantile)*100:.0f}% by importance "
          f"(>= {thr:.4f}): {len(l2f):,} edges")
    if "n_seeds" in l2f.columns:
        rep = float((l2f["n_seeds"] >= 2).mean())
        print(f"    of which reproduced in both seeds: {rep*100:.1f}%")

    # ── name bridge ──────────────────────────────────────────────────────────
    mature = sorted(l1["mirna"].unique())
    br = build_bridge(loci, mature)
    matched_genes = br["symbol"].nunique()
    matched_mature = br["mature"].nunique()
    print(f"  name bridge: {matched_genes:,} miRNA genes <-> "
          f"{matched_mature:,}/{len(mature):,} mature names "
          f"({matched_mature/len(mature)*100:.1f}%)")
    if matched_mature < 0.3 * len(mature):
        raise SystemExit(
            f"{network}: only {matched_mature}/{len(mature)} mature miRNA names "
            f"bridged to a locus -- the ID spaces are not joining, refusing to "
            f"emit an under-populated network")

    # ── TF -> miRNA (through the host gene) ──────────────────────────────────
    # A TF edge onto a host gene is evidence for regulation of the miRNA only
    # when the miRNA is same-strand inside that host (intragenic). Antisense
    # containment does not imply co-transcription, so those rows are KEPT but
    # labelled, never silently pooled.
    hb = br[["symbol", "host_symbol", "host_type", "mature"]].dropna()
    tf2mir = (l2f.merge(hb, left_on="target", right_on="host_symbol",
                        how="inner")
                 .rename(columns={"tf": "TF", "mature": "miRNA",
                                  "target": "host_gene"})
                 [["TF", "miRNA", "host_gene", "host_type", "importance"]])
    if "n_seeds" in l2f.columns:
        tf2mir = tf2mir.merge(
            l2f[["tf", "target", "n_seeds"]].rename(
                columns={"tf": "TF", "target": "host_gene"}),
            on=["TF", "host_gene"], how="left")
    tf2mir = tf2mir.drop_duplicates()
    intra = tf2mir[tf2mir["host_type"] == "intragenic"]
    print(f"  TF->miRNA: {len(tf2mir):,} edges "
          f"({len(intra):,} intragenic, {len(tf2mir)-len(intra):,} antisense)")

    # ── miRNA -> TF (L1 edges whose target is a TF) ──────────────────────────
    tfs = set(pd.read_csv(REF / "tf_list.tsv", sep="\t")["TF"].astype(str))
    mir2tf = (l1[l1["target_symbol"].isin(tfs)]
              .rename(columns={"target_symbol": "TF", "mirna": "miRNA"})
              [["miRNA", "TF", "coef"]])
    print(f"  miRNA->TF: {len(mir2tf):,} edges "
          f"({mir2tf['TF'].nunique():,} TFs, {mir2tf['miRNA'].nunique():,} miRNAs)")

    # ── feedback: both directions present ────────────────────────────────────
    # Restricted to INTRAGENIC tf2mir rows only. Antisense containment does not
    # imply co-transcription (see comment above), so an antisense row is not
    # evidence that the TF regulates the miRNA -- pooling it into "feedback"
    # or an FFL triad would assert a transcriptional edge the data don't
    # support. Flagged by the figures agent (agent_communication/README.md,
    # 2026-09-03) after finding antisense pairs (e.g. CTBP2<->hsa-miR-378d via
    # PDP1) leaking into feedback_healthy_pooled.tsv / ffl_triads_healthy_pooled.tsv.
    # tf2mir itself (the raw edge table written to tf_to_mirna_<network>.tsv)
    # keeps both, labelled via host_type, for anyone who wants the antisense rows.
    fb = intra.merge(mir2tf, on=["TF", "miRNA"], how="inner",
                      suffixes=("_tf2mir", "_mir2tf"))
    fb = fb.drop_duplicates(subset=["TF", "miRNA", "host_gene"])
    print(f"  feedback pairs: {len(fb):,} "
          f"({fb[['TF','miRNA']].drop_duplicates().shape[0]:,} distinct TF-miRNA)")

    # ── FFL triads: TF->miRNA, TF->target, miRNA->target ─────────────────────
    tf_tgt = l2f.rename(columns={"tf": "TF", "target": "target"})[
        ["TF", "target", "importance"]]
    mir_tgt = l1.rename(columns={"target_symbol": "target", "mirna": "miRNA"})[
        ["miRNA", "target", "coef"]]
    ffl = (intra[["TF", "miRNA", "host_gene", "host_type"]]
           .merge(tf_tgt, on="TF", how="inner")
           .merge(mir_tgt, on=["miRNA", "target"], how="inner"))
    # A triad whose target IS the miRNA's own host gene is the TF->miRNA edge
    # restated, not a feed-forward loop.
    ffl = ffl[ffl["target"] != ffl["host_gene"]]
    ffl = ffl.drop_duplicates(subset=["TF", "miRNA", "target"])
    print(f"  FFL triads: {len(ffl):,} "
          f"({ffl['TF'].nunique():,} TFs, {ffl['miRNA'].nunique():,} miRNAs, "
          f"{ffl['target'].nunique():,} targets)")

    for name, df in [("tf_to_mirna", tf2mir), ("mirna_to_tf", mir2tf),
                     ("feedback", fb), ("ffl_triads", ffl)]:
        p = OUT / f"{name}_{network}.tsv"
        df.to_csv(p, sep="\t", index=False)
        prov.record_output(f"{name}_{network}", path=p, rows=len(df))

    (OUT / f"summary_{network}.json").write_text(json.dumps({
        "network": network, "l2_quantile": l2_quantile,
        "l2_importance_threshold": float(thr),
        "l2_edges_used": int(len(l2f)),
        "bridge_mature_matched": int(matched_mature),
        "bridge_mature_total": int(len(mature)),
        "n_tf_to_mirna": int(len(tf2mir)),
        "n_tf_to_mirna_intragenic": int(len(intra)),
        "n_mirna_to_tf": int(len(mir2tf)),
        "n_feedback": int(len(fb)),
        "n_ffl_triads": int(len(ffl)),
        "note": ("L2 filtered to the top importance decile because seed "
                 "reproducibility is 0.761 there vs 0.265 overall"),
        "antisense_filter_note": (
            "feedback and ffl_triads are built from intragenic-only tf2mir "
            "rows (antisense host-containment excluded, since it does not "
            "imply co-transcription). tf_to_mirna_<network>.tsv itself still "
            "carries both, labelled by host_type."),
    }, indent=2))


def main(networks=None, l2_quantile=0.9, min_l2_edges=1000):
    avail = sorted({p.name.split("l1_")[1].rsplit("_full.tsv", 1)[0]
                    for p in L1D.glob("l1_*_full.tsv")})
    if networks is not None and len(networks) == 0:
        raise SystemExit("--networks given with zero values; omit it to mean all")
    chosen = avail if networks is None else list(networks)
    missing = [n for n in chosen if n not in avail]
    if missing:
        raise SystemExit(f"no L1 for: {missing}")
    print(f"=== stage 04: network assembly ===\n  networks: {len(chosen)}")
    for n in chosen:
        assemble(n, l2_quantile, min_l2_edges)
    print("\ndone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="*", default=None)
    ap.add_argument("--l2-quantile", type=float, default=0.9,
                    help="keep L2 edges at/above this importance quantile "
                         "(0.9 = top decile, where seed reproducibility is 0.76)")
    ap.add_argument("--min-l2-edges", type=int, default=1000)
    a = ap.parse_args()
    main(networks=a.networks, l2_quantile=a.l2_quantile,
         min_l2_edges=a.min_l2_edges)
