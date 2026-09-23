# Integrating gene regulation: TF–miRNA co-regulatory networks across TCGA

Which transcription factors regulate miRNA transcription, which miRNAs repress
transcription factors, and which pairs do both across
TCGA cancer types and for pooled healthy tissue.

The pipeline assembles two inferred evidence layers into one directed graph per
cohort, then asks whether the resulting feedback pairs and feed-forward loops
are shared across cancers or cohort-specific.

| Layer | Edge | How it is obtained |
|-------|------|--------------------|
| L1 | miRNA → gene / TF | candidate miRNA–target interactions, filtered per cohort by an elastic-net fit against expression (the SPONGE step-1 criterion: non-positive coefficient) |
| L2 | TF → gene | GRNBoost2 gene regulatory network per cohort, all expressed genes as targets |

TF → miRNA edges are inferred by bridging L2 through miRNA host-gene loci:
1,688 of 1,878 miRNA loci (89.9%) sit inside a host gene, so a TF → host-gene
edge is read as a TF → miRNA edge for intragenic miRNAs. Antisense containment
is kept as a separate class and **not** treated as co-transcription.

**23 networks** are built: 22 primary-tumour cohorts and one `healthy_pooled`
(solid-tissue normal, n = 654), each with at least 125 samples.

## Setup

```bash
git clone <this repo> && cd integration_gene_regulation
export COREG_BASE="$PWD"                              # repo root
export SPONGE_RESOURCES=/path/to/SPONGE_resources     # only for s02a (R)

conda env create -f environment.yml && conda activate grn
```

`COREG_BASE` is optional. If not provided, it falls back to the script location.
`SPONGE_RESOURCES` has no default; `s02a` stops with an explanatory error if it
is unset. The single R script needs only base R (≥ 4.0); it reads `.rda`
resource files and writes a TSV.

## Stages

Run from `coreg_pipeline/`. Each stage writes into `results/` and records its
inputs (URL, checksum, dimensions, path relative to the repository root) into
`coreg_pipeline/manifest.json` via `provenance.py`; `python provenance.py`
prints an audit separating fetched from locally supplied inputs. The manifest
is a per-run log and is not versioned.

| Stage | Command | Writes | Cost |
|---|---|---|---|
| 00 | `python s00_reference.py` | TF list + signs, miRNA loci and host genes | 24 s |
| 01 | `python s01_expression.py --mode all` | `data/expression/<cohort>/{gene,mirna}_log2.parquet`, `samples.tsv`, `meta.json` | 26 min, 349 MB |
| 02 | `python s02_layer1.py --mode full --candidates <candidate_table.tsv>` | `results/layer1/` | 2 min healthy, 45 min all tumour |
| 03 | `./run_s03.sh` (or `./run_s03_tumour.sh`, which reads `nets_tumour.txt`) | `results/layer2/` | ~16 min per network per seed |
| 04 | `python s04_network.py --l2-quantile 0.9` | `results/network/`, `network_summary_23cohorts.tsv` | minutes |
| 05 | `python s05_effects.py --crossfit` | `results/effects/`, `effects_summary_23networks.tsv` | minutes |

Both expression matrices are **already log2** (`meta.json` carries
`already_log2: true`); downstream stages must not transform again.

Stage 02 takes the candidate miRNA–target table as an argument, that was previously assembled from several databases (See below).

### Figures

```bash
python sX_make_figures.py [--only fig2C_cross_cancer_jaccard_and_rewiring]
python sX_make_poster_findings.py [--only ...]
python sX_extract_tumor_specific_circuits.py
python fig_pairwise_jaccard_all_edges.py
python plot_finding_feedback_convergence.py
```

## What is in this repository

Only the scripts that produce the final output, plus the small tables they read
or summarise:

```
coreg_pipeline/
  provenance.py                          input manifest / checksums
  grnboost_local.py                      dask-free GRNBoost2 driver (caveat 5)
  s00_reference.py … s05_effects.py      stages 00-05 (main tree)
  run_s03.sh, run_s03_tumour.sh          stage 03 drivers
  s02a_build_canonical_candidates.R      canonical candidate pool
  s02_layer1_canonical.py                canonical L1
  s04_network_canonical.py               canonical network assembly
  sX_make_figures.py                     publication figures
  sX_make_poster_findings.py             poster figures
  sX_extract_tumor_specific_circuits.py  tumour-specific circuit figures
  fig_pairwise_jaccard_all_edges.py      cross-cohort Jaccard figure
  plot_finding_feedback_convergence.py   feedback-convergence figure
  nets_tumour.txt, networks_22cohorts.txt        cohort lists
  data/reference/{tf_list,tf_signs,mirna_loci}.tsv
  network_summary_23cohorts.tsv, effects_summary_23networks.tsv,
  cross_network_jaccard_23.tsv                   headline summaries
  results_canonical/*.tsv                        canonical summary matrices
mirna_mapping.csv                                miRNA ID map, read by s02a
```

`mirna_loci.tsv` is **long format — one row per (locus, host gene) pair**:
2,333 rows cover 1,878 unique loci, so row and locus counts must not be quoted
interchangeably. Membership of `tf_list.tsv` requires at least one curated
CollecTRI target, so it is precision-weighted, not complete — absence is not
evidence that a gene is not a TF.

## What is not in this repository

Expression matrices, per-cohort L1/L2 tables, GRN outputs, figures, logs and
caches are not versioned but can be fetched by the stages above from public sources:

- **TCGA PanCanAtlas** expression, miRNA and phenotype tables — UCSC Xena
  (`EB++AdjustPANCAN`, batch-corrected, already log2). Bulk download of the
  gene matrix is often blocked; `s01` pulls it in gene batches through the
  `fetch` API instead.
- **miRNA–target candidates** — a merge of lncBase, microT (high confidence),
  tarBase and TargetScan, supplied to `s02` as a file; or the four canonical
  SPONGE resources via `s02a`.
- **TF list and signs** — OmniPath / CollecTRI.
- **miRNA loci and host genes** — Ensembl GTF (release discovered at run time,
  not pinned).

## Caveats to read before using the outputs

1. **GRNBoost2 is not reproducible edge-by-edge.** Two seeds on the identical
   matrix agree at Jaccard 0.265 over the full edge set (983,843 of 3,713,826
   edges for the healthy reference), but reproducibility is strongly
   importance-dependent: 0.761 in the strongest importance decile, 0.050 in the
   weakest. This is the technical noise floor for every downstream claim. Do
   not make single-edge claims from one run; use a multi-seed consensus or
   restrict to the top decile (`--l2-quantile 0.9`, the default in stage 04).
2. **No covariate adjustment.** The correlation and elastic-net steps do not
   adjust for sex, age, batch or tumour purity beyond the batch correction
   already applied to the Xena matrices.
3. **Glioblastoma is excluded.** Its 594 tumour barcodes are present in the
   phenotype table but absent from the miRNA matrix. A cancer type with too few
   miRNA-carrying barcodes is skipped with a message, not fatally.
