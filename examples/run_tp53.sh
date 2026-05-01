#!/usr/bin/env bash
set -euo pipefail

starmap nmf TP53 --coord-csv data/3Dcoord_allgenes.csv --output-dir output/nmf
starmap gistar-legacy TP53
starmap download-distances P04637
starmap annotate-auc-cluster P04637
starmap run-mlp P04637
starmap logodds P04637 --mlp-root output/MLP_outputs --nmf-root output/nmf --mapping-csv data/uniprot_gene_map.csv --output-root output/logodds_results
