#!/bin/bash
#SBATCH --job-name=starmap_nmf
#SBATCH --output=logs/starmap_nmf_%A_%a.out
#SBATCH --error=logs/starmap_nmf_%A_%a.err
#SBATCH --array=1-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
module load anaconda || true
conda activate starmap

GENE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" genes.txt)
starmap nmf "$GENE" --coord-csv data/3Dcoord_allgenes.csv --output-dir output/nmf
