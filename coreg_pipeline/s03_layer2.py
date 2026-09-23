#!/usr/bin/env python
"""
Stage 03 -- Layer 2: TF -> gene edges via GRNBoost2.

TF -> gene is inferred here; the TF -> miRNA bridge (through miRNA host genes)
is stage 04's job, because it is a join on the loci table rather than an
inference step.

TARGETS. All expressed genes, not just annotated miRNA host genes. Restricting
targets to host genes is exact for those targets -- GRNBoost2 fits each target
independently -- but it forecloses the co-regulated-target analysis and the
feed-forward-loop layer, which need TF edges onto ordinary protein-coding
targets too. The previous pipeline's host-gene-only runs covered ~776 targets
per cohort against ~15-18k here.

STOCHASTICITY. GRNBoost2 is a tree ensemble with random feature subsampling, so
edge sets differ between seeds. Measured on the previous data, the Jaccard
similarity between two seeds on the SAME matrix was 0.398 -- meaning roughly
60% of top edges are not reproduced by a rerun. That is a reproducibility
ceiling for every downstream statement about individual edges, so:

  * the seed is recorded in every output's metadata;
  * `--seeds N` runs N seeds and writes a consensus table with the per-edge
    recurrence count, letting a downstream stage require an edge to appear in
    several runs instead of trusting one;
  * a single-seed run is still permitted (it is what the compute budget allows
    for 32 networks) but is flagged in the metadata as `consensus=False`.

Any claim about a SPECIFIC TF->target edge should come from a consensus run.
Aggregate network statistics are far more stable than individual edges.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as prov  # noqa: E402

BASE = Path(__file__).resolve().parent
EXPR = BASE / "data" / "expression"
REF = BASE / "data" / "reference"
OUT = BASE / "results" / "layer2"
OUT.mkdir(parents=True, exist_ok=True)

MIN_EXPR_FRAC = 0.2
N_ESTIMATORS = 5000


def load_network(network: str):
    d = EXPR / network
    gene = pd.read_parquet(d / "gene_log2.parquet")
    meta = json.loads((d / "meta.json").read_text())
    return gene, meta


def expressed_genes(gene: pd.DataFrame) -> list[str]:
    n = gene.shape[1]
    keep = gene.index[(gene > 0).sum(axis=1) >= MIN_EXPR_FRAC * n]
    return sorted(keep)


def run_grnboost(gene: pd.DataFrame, regulators: list[str], targets: list[str],
                 seed: int, n_estimators: int) -> pd.DataFrame:
    """
    One GRNBoost2 run, scheduled with multiprocessing rather than dask.

    arboreto's public entry points build a dask LocalCluster, which cannot start
    in this sandbox; `grnboost_local.grnboost2_mp` runs the identical per-target
    regression under multiprocessing instead (see that module for why the
    importances are unchanged).

    NaNs are filled with the gene's own mean rather than dropped: dropping rows
    would remove a sample from every target's fit, and dropping genes would
    change the target set between networks, making cross-network comparison
    incoherent.
    """
    from grnboost_local import grnboost2_mp

    sub = gene.loc[sorted(set(targets) | set(regulators))]
    mat = sub.T                                     # samples x genes
    mat = mat.fillna(mat.mean(axis=0))
    mat = mat.loc[:, mat.std(axis=0) > 0]           # constant genes carry no signal

    regs = [r for r in regulators if r in mat.columns]
    tgts = [t for t in targets if t in mat.columns]

    net = grnboost2_mp(mat, list(mat.columns), regs, tgts,
                       n_estimators=n_estimators, seed=seed)
    return net.rename(columns={"TF": "tf"})


def process(network: str, tf_all: list[str], seeds: list[int],
            n_estimators: int, max_targets: int | None):
    gene, meta = load_network(network)
    tag = f"l2_{network}"
    print(f"\n=== {network} ({meta['n_samples']} samples, "
          f"{meta['sample_type']}) ===")

    targets = expressed_genes(gene)
    if max_targets:
        targets = targets[:max_targets]
    regs = [t for t in tf_all if t in gene.index]
    print(f"  expressed targets {len(targets):,} | TFs present {len(regs):,}")

    runs = []
    t0 = time.time()
    for s in seeds:
        net = run_grnboost(gene, regs, targets, s, n_estimators)
        net["seed"] = s
        runs.append(net)
        print(f"    seed {s}: {len(net):,} edges ({time.time()-t0:.0f}s)",
              flush=True)

    if len(runs) == 1:
        out_net = runs[0]
        consensus = False
    else:
        # Consensus: per-edge recurrence across seeds plus mean importance.
        cat = pd.concat(runs, ignore_index=True)
        out_net = (cat.groupby(["tf", "target"], as_index=False)
                      .agg(importance=("importance", "mean"),
                           importance_sd=("importance", "std"),
                           n_seeds=("seed", "nunique")))
        consensus = True
        rec = out_net["n_seeds"].value_counts().sort_index().to_dict()
        print(f"  consensus over {len(seeds)} seeds: edge recurrence {rec}")

    out_net["network"] = network
    path = OUT / f"{tag}.tsv"
    out_net.to_csv(path, sep="\t", index=False)

    (OUT / f"{tag}_meta.json").write_text(json.dumps({
        "network": network, "sample_type": meta["sample_type"],
        "n_samples": meta["n_samples"],
        "n_regulators": len(regs), "n_targets": len(targets),
        "n_edges": int(len(out_net)),
        "seeds": seeds, "consensus": consensus,
        "n_estimators": n_estimators,
        "all_expressed_targets": max_targets is None,
        "seed_stability_note": (
            "GRNBoost2 is stochastic; two seeds on the same matrix gave "
            "Jaccard 0.398 on the previous data. Single-seed edge lists should "
            "not be used for claims about individual edges."),
        "input_units": "log2, already transformed by stage 01",
    }, indent=2))
    print(f"  wrote {len(out_net):,} edges -> {path.name}")
    prov.record_output(tag, path=path, rows=len(out_net),
                       params={"seeds": seeds, "consensus": consensus})


def main(networks=None, seeds=(0,), n_estimators=N_ESTIMATORS,
         max_targets=None):
    tf_path = REF / "tf_list.tsv"
    if not tf_path.exists():
        raise SystemExit(f"missing {tf_path}; run s00_reference.py")
    tf_all = pd.read_csv(tf_path, sep="\t")["TF"].astype(str).tolist()
    print(f"=== stage 03: Layer 2 ===\n  TFs available: {len(tf_all):,}")

    avail = sorted(d.name for d in EXPR.iterdir()
                   if d.is_dir() and (d / "meta.json").exists())

    # An EMPTY network list must not silently mean "all networks". A shell
    # driver whose array expanded to nothing passed --networks with zero values;
    # `networks or avail` then treated [] as falsy and ran all 32 networks,
    # including one it was meant to leave alone. Distinguish None (no selection
    # given -> all) from [] (a selection was given and it was empty -> error).
    if networks is not None and len(networks) == 0:
        raise SystemExit(
            "--networks was given with zero values. Refusing to fall back to "
            "all networks: pass names explicitly, use --networks-file, or omit "
            "--networks entirely to mean all.")
    chosen = avail if networks is None else list(networks)
    missing = [n for n in chosen if n not in avail]
    if missing:
        raise SystemExit(f"no expression matrix for: {missing}")
    print(f"  networks: {len(chosen)} | seeds: {list(seeds)}")

    for net in chosen:
        try:
            process(net, tf_all, list(seeds), n_estimators, max_targets)
        except Exception as e:
            # One network failing must not discard the others.
            print(f"  [FAIL] {net}: {type(e).__name__}: {e}", flush=True)
    print("\ndone.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="*", default=None)
    ap.add_argument("--networks-file", default=None,
                    help="file with one network name per line. Preferred over "
                         "--networks for scripted runs: it needs no shell array "
                         "expansion, so names containing shell metacharacters "
                         "cannot split the command")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="one seed per run; >1 writes a consensus table with "
                         "per-edge recurrence counts")
    ap.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    ap.add_argument("--max-targets", type=int, default=None,
                    help="cap target count (smoke tests only)")
    a = ap.parse_args()
    nets = a.networks
    if a.networks_file:
        if nets:
            raise SystemExit("pass either --networks or --networks-file")
        lines = [ln.strip() for ln in Path(a.networks_file).read_text().splitlines()]
        nets = [ln for ln in lines if ln and not ln.startswith("#")]
        if not nets:
            raise SystemExit(f"{a.networks_file} lists no networks")
        print(f"  {len(nets)} networks from {a.networks_file}")
    main(networks=nets, seeds=tuple(a.seeds),
         n_estimators=a.n_estimators, max_targets=a.max_targets)
