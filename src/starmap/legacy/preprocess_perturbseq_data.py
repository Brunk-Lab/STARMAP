# Import necessary packages
import os
import glob
import pandas as pd
import scanpy as sc
import numpy as np
from scipy import sparse as sp
from pathlib import Path

# Load all gene sets 
geneset_dir = "..data/trn_gene_sets/"
gene_sets = {}
for fn in os.listdir(geneset_dir):
    if fn.endswith(".csv"):
        path = os.path.join(geneset_dir, fn)
        genes = pd.read_csv(path, header=None).iloc[:, 0].dropna().astype(str).tolist()
        name = os.path.splitext(fn)[0]
        gene_sets[name] = genes

print(f"Loaded {len(gene_sets)} gene sets")

set_names = list(gene_sets.keys())

# Load PerturbSeq data in backed mode
adata = sc.read_h5ad("../data/downloads/HCT116_filtered_dual_guide_cells.h5ad", backed="r")  

# Build gene→index map
gene_to_idx = {g: i for i, g in enumerate(adata.var_names)}

# Build gene × set weight matrix
rows, cols, vals = [], [], []
for j, s in enumerate(set_names):
    idxs = [gene_to_idx[g] for g in gene_sets[s] if g in gene_to_idx]
    if not idxs:
        continue
    w = 1.0 / len(idxs)  # mean expression weight
    rows.extend(idxs)
    cols.extend([j] * len(idxs))
    vals.extend([w] * len(idxs))

G = sp.csr_matrix((vals, (rows, cols)), shape=(adata.n_vars, len(set_names)))
print(f"Weight matrix built: shape {G.shape}")

# Stream through cells in chunks
outdir = Path("geneset_chunks")
outdir.mkdir(exist_ok=True)

chunk_size = 50_000
for i, start in enumerate(range(0, adata.n_obs, chunk_size)):
    end = min(start + chunk_size, adata.n_obs)
    print(f"Processing cells {start}–{end}...")

    Xc = adata.X[start:end, :]
    Xc = sp.csr_matrix(Xc) if not sp.issparse(Xc) else Xc.tocsr()
    S = Xc.dot(G)  # (cells × sets)

    df = pd.DataFrame(S.toarray(), index=adata.obs_names[start:end], columns=set_names)
    df.reset_index(names="cell_id").to_parquet(
        outdir / f"geneset_means_chunk{i}.parquet",
        index=False,
        compression="zstd",
        engine="pyarrow"
    )

print("Finished writing chunk files.")

# Merge into one final file
print("Merging chunk files...")
all_chunks = [pd.read_parquet(f) for f in sorted(outdir.glob("geneset_means_chunk*.parquet"))]
scores = pd.concat(all_chunks, ignore_index=True)
scores.to_parquet("geneset_means.parquet", compression="zstd", engine="pyarrow")
print(f"Final file written: geneset_means.parquet with shape {scores.shape}")