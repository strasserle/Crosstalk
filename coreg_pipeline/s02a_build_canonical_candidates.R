#!/usr/bin/env Rscript
# s02a_build_canonical_candidates.R
#
# Genome-wide rebuild of the candidate miRNA-target interaction table, using
# the four canonical SPONGE resources (miRTarBase v7, TargetScan, LncBase v2,
# mircode) instead of this project's own merged_interactions_with_sources.tsv
# (lncBase + microT_high + tarBase + targetScan). Unlike the earlier BRCA/
# healthy top-1000-gene pilot build (s04_build_canonical_mir_interactions.R),
# this is NOT restricted to any gene subset -- it covers every gene present
# in any of the 4 resource matrices, matching the scope s02_layer1.py expects
# (candidate = "at least one source predicts this pair"; the elastic-net
# filter downstream does the confidence filtering).
#
# Output: coreg_pipeline/results_canonical/canonical_mir_gene_candidates.tsv
#   columns: target_symbol, mirna, sources, n_sources  (drop-in replacement
#   input format for s02_layer1.py's load_candidates(), which will recompute
#   n_sources itself but we compute it here too for a quick sanity check).

# --- portable path resolution (repo root; override with COREG_BASE) ---
.args <- commandArgs(trailingOnly = FALSE)
.f    <- grep("^--file=", .args, value = TRUE)
.here <- if (length(.f)) dirname(normalizePath(sub("^--file=", "", .f[1]))) else getwd()
BASE <- Sys.getenv("COREG_BASE", unset = normalizePath(file.path(.here, "..")))
RES <- Sys.getenv("SPONGE_RESOURCES", unset = NA_character_)
if (is.na(RES)) stop("Set SPONGE_RESOURCES to the directory holding the canonical SPONGE resource files (miRTarBase v7, TargetScan, LncBase v2, mircode) -- see README.")

OUT  <- file.path(BASE, "coreg_pipeline/results_canonical")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

load_sym <- function(rdata_file, obj_name) {
  e <- new.env()
  load(file.path(RES, rdata_file), envir = e)
  get(obj_name, envir = e)
}

mats <- list(
  mirtarbase = load_sym("mirtarbase7.RData", "mirtarbase_symbol"),
  targetscan = load_sym("targetscan.RData",  "targetscan_symbol"),
  lncbase    = load_sym("lncbase_v2.RData",  "lncbase_symbol"),
  mircode    = load_sym("mircode.RData",     "mircode_symbol")
)

for (nm in names(mats)) {
  m <- mats[[nm]]
  cat(sprintf("  %-10s: %d genes x %d miRNA(MIMAT) cols\n", nm, nrow(m), ncol(m)))
}

# MIMAT -> mature miRNA name bridge (project's own mapping, space-delimited)
map_df <- read.table(file.path(BASE, "mirna_mapping.csv"), header = TRUE, sep = " ",
                      stringsAsFactors = FALSE)
cat("mirna_mapping.csv rows:", nrow(map_df), "\n")
mimat_to_name <- setNames(map_df$Name, map_df$MIMAT_ID)

for (nm in names(mats)) {
  m <- mats[[nm]]
  mapped <- mimat_to_name[colnames(m)]
  n_mapped <- sum(!is.na(mapped))
  cat(sprintf("  %-10s: %d / %d MIMAT columns map to a known mature-miRNA name\n",
              nm, n_mapped, ncol(m)))
  keep <- !is.na(mapped)
  m <- m[, keep, drop = FALSE]
  colnames(m) <- mapped[keep]
  mats[[nm]] <- m
}

# Build long (gene, mir, source) triples from nonzero entries -- sparse, no
# dense union matrix needed (union across genes would be huge x sparse).
per_source_pairs <- list()
for (nm in names(mats)) {
  m <- mats[[nm]]
  idx <- which(m != 0, arr.ind = TRUE)
  if (nrow(idx) > 0) {
    pairs <- data.frame(gene = rownames(m)[idx[, 1]],
                         mir  = colnames(m)[idx[, 2]],
                         source = nm, stringsAsFactors = FALSE)
    per_source_pairs[[nm]] <- pairs
  } else {
    per_source_pairs[[nm]] <- data.frame(gene = character(0), mir = character(0),
                                          source = character(0))
  }
  cat(sprintf("  %-10s: %d (gene,mir) pairs\n", nm, nrow(per_source_pairs[[nm]])))
}

all_pairs <- do.call(rbind, per_source_pairs)
cat("total (gene,mir,source) rows across 4 sources:", nrow(all_pairs), "\n")

# Aggregate to one row per (gene, mir) with a comma-joined sources list
agg <- aggregate(source ~ gene + mir, data = all_pairs,
                  FUN = function(s) paste(sort(unique(s)), collapse = ","))
agg$n_sources <- sapply(strsplit(agg$source, ","), length)
names(agg) <- c("target_symbol", "mirna", "sources", "n_sources")

cat("unique (gene,mir) candidate pairs (union):", nrow(agg), "\n")
cat("n_sources distribution:\n")
print(table(agg$n_sources))
cat("unique genes:", length(unique(agg$target_symbol)), "\n")
cat("unique miRNAs:", length(unique(agg$mirna)), "\n")

write.table(agg, file.path(OUT, "canonical_mir_gene_candidates.tsv"),
            sep = "\t", row.names = FALSE, quote = FALSE)
cat("saved:", file.path(OUT, "canonical_mir_gene_candidates.tsv"), "\n")
