# Import necessary packages
import sys
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Check input
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py GENE")
    sys.exit(1)

target_gene = sys.argv[1].upper().strip()

# Load data
scores = pd.read_parquet("../output/geneset_means.parquet")  # columns: cell_id + pathways
adata = sc.read_h5ad("../data/downloads/HCT116_filtered_dual_guide_cells.h5ad", backed="r")
meta = adata.obs.reset_index(names="cell_id")

# Merge scores + metadata
df = scores.merge(meta, on="cell_id")

# Define meta vs pathway columns
meta_cols = [
    "cell_id", "sample", "num_features", "guide_target", "gene_target",
    "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt",
    "pass_guide_filter"
]
set_names = [c for c in df.columns if c not in meta_cols]

# Convert to NumPy arrays for speed
X = df[set_names].to_numpy()
gene_labels = df["gene_target"].values
mask = df["pass_guide_filter"].values

# Precompute indices for each group
groups = {g: np.where((gene_labels == g) & mask)[0] for g in np.unique(gene_labels)}
non_target_idx = groups.get("non-targeting", None)

# Process only requested gene
if target_gene not in groups:
    print(f"Gene '{target_gene}' not found in gene_target column.")
    sys.exit(1)

if target_gene == "non-targeting":
    print("Please provide a perturbed gene, not 'non-targeting'.")
    sys.exit(1)

pert_idx = groups[target_gene]

# Define controls
ctrl_idx = non_target_idx if non_target_idx is not None else np.where((gene_labels != target_gene) & mask)[0]

if len(pert_idx) < 5 or len(ctrl_idx) < 5:
    print(f"Not enough cells for {target_gene}: {len(pert_idx)} perturbed, {len(ctrl_idx)} controls")
    sys.exit(1)

pert = X[pert_idx, :]
ctrl = X[ctrl_idx, :]

# Vectorized stats across all pathways
delta = pert.mean(axis=0) - ctrl.mean(axis=0)
_, pvals = ttest_ind(pert, ctrl, axis=0, equal_var=False)

df_out = pd.DataFrame({
    "gene": target_gene,
    "pathway": set_names,
    "delta": delta,
    "pval": pvals,
    "n_pert": len(pert_idx),
    "n_ctrl": len(ctrl_idx),
})

# Save output
out_file = f"../output/{target_gene}_perturbed_df.csv"
df_out.to_csv(out_file, index=False)

print(f"Finished {target_gene} ({len(pert_idx)} perturbed, {len(ctrl_idx)} controls)")
print(f"Saved to {out_file}")