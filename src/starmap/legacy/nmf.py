# Import necessary packages
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.manifold import MDS
from scipy.spatial.distance import squareform, pdist
from sklearn.cluster import KMeans
import sys
import os

# Config
gene_3D_data = '../data/3Dcoord_allgenes.csv'

# Functions
def min_max_scaling(column):
    return (column - column.min()) / (column.max() - column.min())


# Model initialization
mds = MDS(n_components=2, 
          max_iter=3000, 
          eps=1e-9,                 
          dissimilarity="precomputed",  
          n_jobs=1,                 
          random_state=1
         )

# Gene name from command line
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py GENE")
    sys.exit(1)

gene = sys.argv[1].upper().strip()

# Create an output folder named by the first letter of the gene
os.makedirs(f'../output/{gene[0]}', exist_ok=True)

# Load data
info = pd.read_csv(gene_3D_data)

# Subset to the selected gene
df = info[info['gene'] == gene]
if df.empty:
    raise ValueError(f"No data found for gene: {gene}")

# Select coordinate columns, set residue IDs as the index
GE = df[['x_coord', 'y_coord', 'z_coord', 'res']].set_index('res')
ge_ex = GE.copy()

# Drop columns with constant values (no variance)
ge_ex = ge_ex.loc[:, ge_ex.nunique() > 1]

# Convert to numeric and perform rank-mean normalization
ge_ex_mut = ge_ex.apply(pd.to_numeric)
rank_mean = ge_ex_mut.stack().groupby(
    ge_ex_mut.rank(method='first').stack().astype(int)
).mean()
ge_ex_mut_ranked = (
    ge_ex_mut.rank(method='min').stack().astype(int).map(rank_mean).unstack()
)

# Apply min–max scaling per column so all features are on a comparable scale
for col in ge_ex_mut_ranked.columns:
    ge_ex_mut_ranked[col] = min_max_scaling(ge_ex_mut_ranked[col])

# This scaled, normalized matrix serves as input for NMF
V = ge_ex_mut_ranked

# NMF Component Search

# Find the optimal number of NMF components
best_n = 3
min_error = float('inf')

for k in range(3, 7):
    model = NMF(n_components=k, init='random', random_state=0, max_iter=1000)
    W = model.fit_transform(V)        # Gene × Component matrix
    H = model.components_             # Component × Feature matrix
    mse = mean_squared_error(V, W @ H)
    if mse < min_error:
        min_error = mse
        best_n = k

# Final NMF
# Fit final model with optimal component number
model = NMF(n_components=best_n, init='random', random_state=0)
W = model.fit_transform(V)

# Assign component labels (C1, C2, C3, ...)
indices = [f'C{i+1}' for i in range(best_n)]
w_df = pd.DataFrame(W, columns=indices, index=GE.index).clip(lower=-3.25, upper=3.25)

# Compute 'altitude' as an inverse mean activation across components
work_df = w_df.copy()
work_df['altitude'] = 1 - work_df[indices].mean(axis=1)
work_df = work_df.astype(float)

# MDS + clustering
df_mds = work_df.apply(min_max_scaling)
background_similarities = squareform(pdist(df_mds, 'euclidean'))
pos = mds.fit(background_similarities).embedding_
df_pos = pd.DataFrame(pos, columns=['x_axis', 'y_axis'], index=df_mds.index)

# Normalize axes and add altitude column
for col in ['x_axis', 'y_axis']:
    df_pos[col] = min_max_scaling(df_pos[col])
df_pos['altitude'] = df_mds['altitude']

# Perform KMeans clustering in 2D embedding space
model = KMeans(n_clusters=best_n, random_state=0)
df_pos['clust'] = model.fit_predict(df_pos[['x_axis', 'y_axis']])

# Sort by cluster for cleaner output
df_pos = df_pos.sort_values(by='clust')

# Save final dataframe containing x/y coordinates, altitude, and cluster labels
out_path = f'../output/{gene[0]}/{gene}_nmfinfo_final.csv'
df_pos.to_csv(out_path)

print(f"{gene} IS DONE, saved to {out_path}")