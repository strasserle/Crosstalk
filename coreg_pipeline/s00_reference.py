#!/usr/bin/env python
"""
Stage 00 -- reference data, all fetched from public APIs.

Produces three reference tables:

  tf_list.tsv        Transcription factors, taken as the set of TFs carrying at
                     least one curated target in OmniPath/CollecTRI. This
                     deliberately replaces the permissive SCENIC allTFs list
                     used previously, which contained non-TFs (HNRNPUL1, RBM8A,
                     PRKAA2, TMEM33, AGMAT all appeared as network hubs).

                     CAVEAT on this definition: membership requires a curated
                     TF->target interaction, so a genuine TF with no curated
                     targets is absent. The list is therefore precision-weighted
                     rather than complete, which is the right trade for hub
                     ranking (the previous failure was false positives) but
                     means absence from tf_list.tsv is not evidence that a gene
                     is not a TF. No GO or motif evidence is combined in.

  tf_signs.tsv       Per-TF dominant regulatory sign (activator/repressor/mixed)
                     from CollecTRI, used to classify circuit topology.

  mirna_loci.tsv     miRNA gene loci with coordinates and host-gene assignment
                     from Ensembl. This is what bridges TF->gene edges to
                     TF->miRNA edges: a TF regulating a miRNA's host gene (or a
                     gene whose span contains the miRNA) is evidence for it
                     regulating the miRNA.

Nothing here reads a user-supplied file.
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

import pandas as pd
import requests

# This environment's interpreter starts with isolated path semantics: neither the
# cwd nor the script's own directory lands on sys.path, so a plain
# `import provenance` fails even though the module sits alongside this file.
# Inserting the script directory explicitly makes every stage runnable as
# `python sNN_*.py` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import provenance as prov  # noqa: E402

BASE = Path(__file__).resolve().parent
REF = BASE / "data" / "reference"
REF.mkdir(parents=True, exist_ok=True)

OMNIPATH = "https://omnipathdb.org/interactions"
ENSEMBL = "https://rest.ensembl.org"
HDR = {"Accept": "application/json"}


# ── TF list + signs ──────────────────────────────────────────────────────────
def fetch_collectri() -> pd.DataFrame:
    """CollecTRI TF->target with signs, gene symbols resolved server-side."""
    url = f"{OMNIPATH}?datasets=collectri&format=tsv&genesymbols=1"
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="\t")
    prov.record_input("collectri", url=url, rows=len(df), cols=df.shape[1],
                      note="TF->target interactions with consensus signs")
    return df


def build_tf_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ctr = fetch_collectri()
    directed = ctr[ctr["is_directed"] == 1] if "is_directed" in ctr else ctr

    sign = (directed.groupby("source_genesymbol")
            .agg(n_activating=("is_stimulation", "sum"),
                 n_repressing=("is_inhibition", "sum"),
                 n_targets=("target_genesymbol", "nunique"))
            .reset_index().rename(columns={"source_genesymbol": "TF"}))
    sign["dominant_sign"] = [
        "activator" if a > r else ("repressor" if r > a else "mixed")
        for a, r in zip(sign["n_activating"], sign["n_repressing"])]

    tfs = sign[["TF", "n_targets", "dominant_sign"]].copy()
    tfs["source"] = "collectri"
    tfs = tfs[tfs["TF"].notna() & (tfs["TF"].astype(str).str.len() > 0)]
    tfs = tfs.sort_values("n_targets", ascending=False).reset_index(drop=True)
    return tfs, sign


# ── miRNA loci ───────────────────────────────────────────────────────────────
ENSEMBL_FTP = "https://ftp.ensembl.org/pub/"


def discover_gtf() -> tuple[str, str]:
    """
    Find the newest Ensembl human GTF, returning (url, filename).

    Releases are walked newest-first rather than taking the highest number
    outright: the newest directory can exist while still being published, so a
    release whose gtf/homo_sapiens/ listing has no GRCh38 GTF is skipped rather
    than treated as a hard failure. `current_gtf/` is not used because it 404s
    on this mirror.
    """
    r = requests.get(ENSEMBL_FTP, timeout=180)
    r.raise_for_status()
    releases = sorted({int(m) for m in re.findall(r'href="release-(\d+)/"', r.text)},
                      reverse=True)
    if not releases:
        raise SystemExit(f"no release-N directories at {ENSEMBL_FTP}")
    for rel in releases:
        d = f"{ENSEMBL_FTP}release-{rel}/gtf/homo_sapiens/"
        try:
            rr = requests.get(d, timeout=180)
            if rr.status_code != 200:
                continue
            names = re.findall(r'href="(Homo_sapiens\.GRCh38\.\d+\.gtf\.gz)"', rr.text)
            if names:
                name = sorted(set(names))[-1]
                print(f"  Ensembl release {rel}: {name}")
                return d + name, name
        except Exception:
            continue
    raise SystemExit(f"no GRCh38 GTF found in releases {releases[:5]}")


def fetch_mirna_loci() -> pd.DataFrame:
    """
    All Ensembl human genes with coordinates and biotype, parsed from the GTF.

    A GTF download is used rather than ~620 paged /overlap/region REST calls:
    one transfer instead of hundreds of requests that are individually
    rate-limitable and can fail partially, which would silently truncate the
    gene set and thus silently shrink miRNA host coverage.

    The release is discovered from the directory listing rather than hardcoded,
    so this does not break when Ensembl increments the release number.
    """
    url, gtf_name = discover_gtf()

    local = REF / gtf_name
    if not local.exists():
        with requests.get(url, timeout=1800, stream=True) as r:
            r.raise_for_status()
            with open(local, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    print(f"  downloaded {local.stat().st_size/1e6:.0f} MB")

    rows = []
    with gzip.open(local, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            attr = f[8]
            gid = re.search(r'gene_id "([^"]+)"', attr)
            bt = re.search(r'gene_biotype "([^"]+)"', attr)
            sym = re.search(r'gene_name "([^"]+)"', attr)
            rows.append({"ensg": gid.group(1) if gid else None,
                         "symbol": sym.group(1) if sym else None,
                         "biotype": bt.group(1) if bt else None,
                         "chrom": f[0],
                         "start": int(f[3]),
                         "end": int(f[4]),
                         "strand": 1 if f[6] == "+" else -1})

    allg = (pd.DataFrame(rows)
            .dropna(subset=["ensg"])
            .drop_duplicates(subset="ensg"))
    print(f"  genes parsed: {len(allg):,} "
          f"({(allg.biotype=='miRNA').sum():,} miRNA biotype)")
    prov.record_input("ensembl_gtf", url=url, path=local, rows=len(allg),
                      note=f"gene records from {gtf_name}")
    return allg


def assign_hosts(allg: pd.DataFrame) -> pd.DataFrame:
    """
    For every miRNA locus, find the genes whose span contains it.

    Two host classes are recorded separately because they carry different
    evidence strength:
      intragenic  -- the miRNA sits inside a host gene on the SAME strand, so it
                     is co-transcribed and a TF acting on the host is acting on
                     the miRNA.
      antisense   -- contained but on the opposite strand; co-transcription does
                     NOT follow, so these are kept but flagged.
    """
    mirs = allg[allg["biotype"] == "miRNA"].copy()
    hosts = allg[allg["biotype"] != "miRNA"].copy()
    print(f"  miRNA loci: {len(mirs):,} | candidate host genes: {len(hosts):,}")

    out = []
    for chrom, mg in mirs.groupby("chrom"):
        hc = hosts[hosts["chrom"] == chrom]
        if hc.empty:
            continue
        hs, he = hc["start"].values, hc["end"].values
        for _, m in mg.iterrows():
            hit = (hs <= m["start"]) & (he >= m["end"])
            if not hit.any():
                out.append({**m.to_dict(), "host_symbol": None,
                            "host_ensg": None, "host_type": "intergenic"})
                continue
            for _, h in hc[hit].iterrows():
                out.append({**m.to_dict(), "host_symbol": h["symbol"],
                            "host_ensg": h["ensg"],
                            "host_type": ("intragenic" if h["strand"] == m["strand"]
                                          else "antisense")})
    loci = pd.DataFrame(out)
    return loci


def main(skip_loci=False):
    print("=== stage 00: reference data ===")
    print("\n[1/2] TF list and signs from CollecTRI")
    tfs, signs = build_tf_tables()
    tfs.to_csv(REF / "tf_list.tsv", sep="\t", index=False)
    signs.to_csv(REF / "tf_signs.tsv", sep="\t", index=False)
    print(f"  TFs: {len(tfs):,}  "
          f"({(tfs.dominant_sign=='activator').sum()} act / "
          f"{(tfs.dominant_sign=='repressor').sum()} rep / "
          f"{(tfs.dominant_sign=='mixed').sum()} mixed)")
    prov.record_output("tf_list", path=REF / "tf_list.tsv", rows=len(tfs))
    prov.record_output("tf_signs", path=REF / "tf_signs.tsv", rows=len(signs))

    if skip_loci:
        print("\n[2/2] miRNA loci: SKIPPED (--skip-loci)")
        return
    print("\n[2/2] miRNA loci and host genes from Ensembl")
    allg = fetch_mirna_loci()
    allg.to_csv(REF / "ensembl_genes.tsv", sep="\t", index=False)
    loci = assign_hosts(allg)
    loci.to_csv(REF / "mirna_loci.tsv", sep="\t", index=False)
    n_mir = loci["ensg"].nunique()
    cov = loci[loci["host_type"] != "intergenic"]["ensg"].nunique()
    print(f"  miRNA loci {n_mir:,} | with a host gene {cov:,} ({cov/n_mir*100:.1f}%)")
    print(f"  host_type: {loci['host_type'].value_counts().to_dict()}")
    prov.record_output("mirna_loci", path=REF / "mirna_loci.tsv", rows=len(loci),
                       note=f"{cov}/{n_mir} miRNA loci have a host gene")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-loci", action="store_true",
                    help="only rebuild the TF tables (loci scan is slow)")
    a = ap.parse_args()
    main(skip_loci=a.skip_loci)
