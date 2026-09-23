#!/usr/bin/env python
"""
Stage 01 -- expression matrices from UCSC Xena PanCanAtlas.

Design decisions and why:

SOURCE. PanCanAtlas via Xena rather than per-sample GDC downloads. Two reasons.
  (a) Batch correction. The gene matrix is EB++Adjust, corrected across the
      whole pan-cancer set; the miRNA matrix is likewise batch-adjusted on
      protocol and platform. Pooling normals from many cancer types and then
      comparing types REQUIRES consistent cross-cohort normalisation, which
      per-cohort GDC downloads do not provide.
  (b) Sample type is verifiable. Columns are TCGA barcodes and the phenotype
      table maps every barcode to its sample_type. The previous GDC fetcher
      filtered on no sample type at all and keyed by case_id, so where a case
      had both tumour and normal files one silently overwrote the other and the
      resulting matrix was ~90% tumour with unreproducible contamination.
      Here sample_type is read from a table and asserted, not assumed.

TRANSFER. The bulk gene matrix is not downloadable from this network, but Xena's
  `fetch` query returns arbitrary gene x sample slices at ~1M values/s, so the
  matrix is pulled in gene batches. Only genes in the requested universe are
  transferred, which is also less data than the full matrix.

UNITS. Both matrices are already log2-transformed by Xena (gene: log2(norm+1);
  miRNA: log2(RPM+1)). They are stored as-is and the units recorded in the
  manifest. Downstream stages must NOT log them again -- the pipeline passes
  `already_log2=True` in the output metadata so a consumer can check rather
  than guess.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as prov  # noqa: E402

BASE = Path(__file__).resolve().parent
EXPR = BASE / "data" / "expression"
EXPR.mkdir(parents=True, exist_ok=True)

HUB = "https://pancanatlas.xenahubs.net"
DL = f"{HUB}/download/"
DATA = f"{HUB}/data/"

GENE_TABLE = "EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena"
MIR_FILE = ("pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs"
            "_08_04_16.xena.gz")
PHENO_FILE = "TCGA_phenotype_denseDataOnlyDownload.tsv.gz"

TUMOUR = "Primary Tumor"
NORMAL = "Solid Tissue Normal"


def sanitize(name: str) -> str:
    """
    Disease name -> directory-safe token.

    Everything outside [a-z0-9_] collapses to '_'. This is stricter than
    stripping a few punctuation marks: TCGA disease names contain '&'
    ("cervical & endocervical cancer", "head & neck squamous cell carcinoma",
    "pheochromocytoma & paraganglioma"), and an '&' in a path is read by the
    shell as a background operator. An earlier version kept '&' and a shell
    driver silently split its own command list on it, building zero of 31
    tumour networks while reporting exit 0.
    """
    out = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return re.sub(r"_+", "_", out)


# ── downloads ────────────────────────────────────────────────────────────────
def get_phenotype() -> pd.DataFrame:
    """barcode -> sample_type, cancer type. This is the sample-type authority."""
    path = EXPR / "phenotype.tsv"
    url = DL + PHENO_FILE
    if not path.exists():
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        ph = pd.read_csv(io.BytesIO(gzip.decompress(r.content)), sep="\t")
        ph.to_csv(path, sep="\t", index=False)
    ph = pd.read_csv(path, sep="\t")
    prov.record_input("xena_phenotype", url=url, path=path, rows=len(ph),
                      note="barcode -> sample_type + _primary_disease")
    return ph


def get_mirna() -> pd.DataFrame:
    """Full miRNA x sample matrix (small enough to download whole)."""
    path = EXPR / "mirna_all.parquet"
    url = DL + MIR_FILE
    if not path.exists():
        r = requests.get(url, timeout=900)
        r.raise_for_status()
        m = pd.read_csv(io.BytesIO(gzip.decompress(r.content)), sep="\t",
                        index_col=0)
        # Same defensive coercion as the gene fetch: the distributed file may
        # carry literal "NaN" strings in otherwise numeric columns.
        m = m.apply(pd.to_numeric, errors="coerce")
        m.to_parquet(path)
    m = pd.read_parquet(path)
    prov.record_input("xena_mirna", url=url, path=path,
                      rows=m.shape[0], cols=m.shape[1],
                      note="mature-strand miRNA, log2(RPM+1), batch-adjusted")
    return m


def fetch_genes(genes: list[str], samples: list[str], batch=800,
                retries=4) -> pd.DataFrame:
    """
    Pull a gene x sample block through Xena's fetch API in gene batches.

    Missing genes come back as all-NaN columns rather than raising, so a
    requested symbol absent from the matrix is visible downstream (and counted
    here) instead of silently shrinking the universe.
    """
    out, t0 = [], time.time()
    for i in range(0, len(genes), batch):
        chunk = genes[i:i + batch]
        q = ('(fetch [{:table "%s" :samples %s :columns %s}])'
             % (GENE_TABLE, json.dumps(samples), json.dumps(chunk)))
        for attempt in range(retries):
            try:
                r = requests.post(DATA, data=q, timeout=600,
                                  headers={"Content-Type": "text/plain"})
                r.raise_for_status()
                arr = r.json()[0] if isinstance(r.json(), list) and \
                    len(r.json()) == 1 and isinstance(r.json()[0], list) and \
                    len(r.json()[0]) == len(chunk) else r.json()
                out.append(pd.DataFrame(arr, index=chunk, columns=samples))
                break
            except Exception as e:
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        done = min(i + batch, len(genes))
        el = time.time() - t0
        print(f"    genes {done:,}/{len(genes):,}  {el:.0f}s "
              f"({done*len(samples)/max(el,1e-9)/1e3:.0f}k vals/s)", flush=True)
    mat = pd.concat(out)

    # Xena returns missing values as the STRING "NaN", not a JSON null, so any
    # column containing one arrives as object dtype -- which silently poisons
    # every downstream numeric operation and fails the parquet write. Coerce
    # once here, at the boundary, and assert the result is float.
    mat = mat.apply(pd.to_numeric, errors="coerce")
    bad = [c for c, dt in mat.dtypes.items() if not np.issubdtype(dt, np.floating)]
    if bad:
        raise SystemExit(f"non-numeric columns survived coercion: {bad[:5]}")

    n_allnan = int(mat.isna().all(axis=1).sum())
    print(f"  fetched {mat.shape[0]:,} genes x {mat.shape[1]:,} samples "
          f"({n_allnan:,} genes absent from matrix)")
    return mat


# ── universe ─────────────────────────────────────────────────────────────────
def gene_universe(candidates_path: Path | None, tf_list: Path) -> list[str]:
    """
    Genes to fetch: TFs (needed as regulators) plus miRNA-target candidates
    (needed as targets). Passing a candidates file is optional so the stage can
    run on TFs alone for a smoke test.
    """
    tfs = pd.read_csv(tf_list, sep="\t")["TF"].astype(str).tolist()
    genes = set(tfs)
    if candidates_path and Path(candidates_path).exists():
        c = pd.read_csv(candidates_path, sep="\t", usecols=["target_symbol"])
        genes |= set(c["target_symbol"].dropna().astype(str))
    genes = sorted(g for g in genes if g and g != "nan")
    print(f"  gene universe: {len(genes):,} "
          f"({len(tfs):,} TFs + candidate targets)")
    return genes


# ── main ─────────────────────────────────────────────────────────────────────
def build_matrix(name: str, barcodes: list[str], genes: list[str],
                 mirna_all: pd.DataFrame, sample_type: str,
                 projects: list[str], batch: int, min_samples: int = 1) -> bool:
    """
    Assemble and save one (gene, miRNA) matrix pair for a sample set.

    Returns True if written, False if skipped. A cancer type with no miRNA
    coverage is SKIPPED, not fatal: some TCGA types are absent from the
    PanCanAtlas miRNA matrix entirely (glioblastoma has 594 tumour barcodes in
    the phenotype table and none in the miRNA matrix), and aborting the run for
    one such type would discard every other type built so far.
    """
    shared = [b for b in barcodes if b in mirna_all.columns]
    if len(shared) < min_samples:
        print(f"  [skip] {name}: {len(barcodes):,} barcodes but only "
              f"{len(shared)} have miRNA data (min {min_samples})")
        return False

    out_dir = EXPR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{name}] {len(barcodes):,} barcodes -> {len(shared):,} with miRNA data")

    gm = fetch_genes(genes, shared, batch=batch)
    mm = mirna_all[shared]

    # Drop genes/miRNAs that are entirely missing for this sample set.
    gm = gm[~gm.isna().all(axis=1)]
    mm = mm[~mm.isna().all(axis=1)]

    gm.to_parquet(out_dir / "gene_log2.parquet")
    mm.to_parquet(out_dir / "mirna_log2.parquet")
    meta = {"name": name, "sample_type": sample_type, "projects": projects,
            "n_samples": len(shared), "n_genes": int(gm.shape[0]),
            "n_mirnas": int(mm.shape[0]),
            "units_gene": "log2(norm_count+1), EB++Adjust batch-corrected",
            "units_mirna": "log2(RPM+1), batch-adjusted on protocol+platform",
            "already_log2": True,
            "source": "UCSC Xena PanCanAtlas"}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    pd.DataFrame({"barcode": shared, "sample_type": sample_type}).to_csv(
        out_dir / "samples.tsv", sep="\t", index=False)
    print(f"  [{name}] genes {gm.shape} | miRNAs {mm.shape}")
    prov.record_output(f"expr_{name}", path=out_dir / "gene_log2.parquet",
                       rows=gm.shape[0], cols=gm.shape[1],
                       note=f"{sample_type}; {len(shared)} samples",
                       params={"sample_type": sample_type, "projects": projects})
    return True


def main(mode="all", candidates=None, tf_list=None, batch=800,
         max_samples=None, max_genes=None, types=None, min_samples=20):
    print("=== stage 01: expression ===")
    tf_list = Path(tf_list or BASE / "data" / "reference" / "tf_list.tsv")
    if not tf_list.exists():
        raise SystemExit(f"missing {tf_list}; run s00_reference.py first")

    ph = get_phenotype()
    print(f"  phenotype: {len(ph):,} samples, "
          f"{ph['_primary_disease'].nunique()} diseases")
    print(f"  {ph['sample_type'].value_counts().to_dict()}")

    mirna_all = get_mirna()
    print(f"  miRNA matrix: {mirna_all.shape}")

    genes = gene_universe(candidates, tf_list)
    if max_genes:
        genes = genes[:max_genes]
        print(f"  [test] gene universe truncated to {len(genes):,}")

    # ── coverage report BEFORE fetching anything ─────────────────────────────
    # miRNA coverage decides what is buildable, and it is not uniform across
    # cancer types. Reporting it first means the achievable scope is known in
    # advance rather than discovered by a failure mid-run.
    mir_bc = set(mirna_all.columns)
    cov_rows = []
    for st, lab in ((NORMAL, "healthy"), (TUMOUR, "tumour")):
        sub = ph[ph["sample_type"] == st]
        for disease, grp in sub.groupby("_primary_disease"):
            n_all = len(grp)
            n_mir = sum(1 for b in grp["sample"] if b in mir_bc)
            cov_rows.append({"sample_type": lab, "disease": disease,
                             "n_barcodes": n_all, "n_with_mirna": n_mir})
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(EXPR / "mirna_coverage.tsv", sep="\t", index=False)
    tum_cov = cov[cov["sample_type"] == "tumour"].sort_values("n_with_mirna",
                                                              ascending=False)
    n_zero = int((tum_cov["n_with_mirna"] == 0).sum())
    print(f"\n  miRNA coverage: {len(tum_cov)} tumour types, "
          f"{n_zero} with NO miRNA data")
    if n_zero:
        print("    no-coverage types: "
              f"{tum_cov[tum_cov['n_with_mirna']==0]['disease'].tolist()}")
    print(f"    healthy pooled: "
          f"{int(cov[cov['sample_type']=='healthy']['n_with_mirna'].sum())} "
          f"of {int(cov[cov['sample_type']=='healthy']['n_barcodes'].sum())} "
          f"normals have miRNA data")

    built, skipped = [], []

    # ── pooled healthy reference ─────────────────────────────────────────────
    if mode in ("all", "healthy"):
        norm = ph[ph["sample_type"] == NORMAL]
        bc = norm["sample"].tolist()
        if max_samples:
            bc = bc[:max_samples]
        print(f"\n[healthy] pooled normals across "
              f"{norm['_primary_disease'].nunique()} diseases")
        ok = build_matrix("healthy_pooled", bc, genes, mirna_all, NORMAL,
                          sorted(norm["_primary_disease"].unique().tolist()),
                          batch, min_samples=min_samples)
        (built if ok else skipped).append("healthy_pooled")

    # ── per-cancer-type tumour ───────────────────────────────────────────────
    if mode in ("all", "tumour"):
        tum = ph[ph["sample_type"] == TUMOUR]
        by_type = tum.groupby("_primary_disease")["sample"].apply(list)
        chosen = types or sorted(by_type.index)
        print(f"\n[tumour] {len(chosen)} cancer types requested")
        for disease in chosen:
            if disease not in by_type:
                print(f"  [skip] {disease}: not in phenotype table")
                skipped.append(disease)
                continue
            bc = by_type[disease]
            if max_samples:
                bc = bc[:max_samples]
            safe = sanitize(disease)
            print(f"\n  --- {disease} ({len(bc)} tumour barcodes) ---")
            ok = build_matrix(f"tumour_{safe}", bc, genes, mirna_all, TUMOUR,
                              [disease], batch, min_samples=min_samples)
            (built if ok else skipped).append(disease)

    (EXPR / "build_report.json").write_text(json.dumps(
        {"built": built, "skipped": skipped,
         "min_samples": min_samples}, indent=2))
    print(f"\ndone.  built {len(built)}, skipped {len(skipped)}")
    if skipped:
        print(f"  skipped: {skipped}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "healthy", "tumour"], default="all")
    ap.add_argument("--candidates", default=None,
                    help="miRNA-target candidate table (for the target universe)")
    ap.add_argument("--tf-list", default=None)
    ap.add_argument("--batch", type=int, default=800)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap samples per matrix (smoke tests only)")
    ap.add_argument("--max-genes", type=int, default=None,
                    help="cap gene universe (smoke tests only)")
    ap.add_argument("--types", nargs="*", default=None,
                    help="specific _primary_disease values to build")
    ap.add_argument("--min-samples", type=int, default=20,
                    help="skip a network with fewer than this many samples "
                         "carrying miRNA data (default 20)")
    a = ap.parse_args()
    main(mode=a.mode, candidates=a.candidates, tf_list=a.tf_list,
         batch=a.batch, max_samples=a.max_samples, max_genes=a.max_genes,
         types=a.types, min_samples=a.min_samples)
