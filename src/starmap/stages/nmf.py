from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.manifold import MDS
from sklearn.metrics import mean_squared_error


def min_max_scaling(column: pd.Series) -> pd.Series:
    denom = column.max() - column.min()
    if denom == 0:
        return pd.Series(0.0, index=column.index)
    return (column - column.min()) / denom


def run_nmf_for_gene(
    gene: str,
    coord_csv: str | Path,
    output_dir: str | Path,
    min_components: int = 3,
    max_components: int = 6,
    random_state: int = 0,
) -> Path:
    """Create a 2D NMF/MDS flatmap and cluster assignment file for one gene."""
    gene = gene.upper().strip()
    coord_csv = Path(coord_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = pd.read_csv(coord_csv)
    df = info[info["gene"].astype(str).str.upper() == gene]
    if df.empty:
        raise ValueError(f"No data found for gene: {gene}")

    ge = df[["x_coord", "y_coord", "z_coord", "res"]].set_index("res")
    ge_ex = ge.loc[:, ge.nunique() > 1].apply(pd.to_numeric)

    rank_mean = ge_ex.stack().groupby(ge_ex.rank(method="first").stack().astype(int)).mean()
    ranked = ge_ex.rank(method="min").stack().astype(int).map(rank_mean).unstack()
    ranked = ranked.apply(min_max_scaling)

    best_n = min_components
    min_error = float("inf")
    for k in range(min_components, max_components + 1):
        model = NMF(n_components=k, init="random", random_state=random_state, max_iter=1000)
        w = model.fit_transform(ranked)
        h = model.components_
        mse = mean_squared_error(ranked, w @ h)
        if mse < min_error:
            min_error = mse
            best_n = k

    model = NMF(n_components=best_n, init="random", random_state=random_state, max_iter=1000)
    w = model.fit_transform(ranked)
    indices = [f"C{i + 1}" for i in range(best_n)]
    w_df = pd.DataFrame(w, columns=indices, index=ge.index).clip(lower=-3.25, upper=3.25)
    work_df = w_df.copy()
    work_df["altitude"] = 1 - work_df[indices].mean(axis=1)

    df_mds = work_df.apply(min_max_scaling)
    distances = squareform(pdist(df_mds, "euclidean"))
    mds = MDS(
        n_components=2,
        max_iter=3000,
        eps=1e-9,
        dissimilarity="precomputed",
        n_jobs=1,
        random_state=1,
    )
    pos = mds.fit(distances).embedding_
    df_pos = pd.DataFrame(pos, columns=["x_axis", "y_axis"], index=df_mds.index)
    for col in ["x_axis", "y_axis"]:
        df_pos[col] = min_max_scaling(df_pos[col])
    df_pos["altitude"] = df_mds["altitude"]
    df_pos["clust"] = KMeans(n_clusters=best_n, random_state=random_state).fit_predict(df_pos[["x_axis", "y_axis"]])
    df_pos = df_pos.sort_values(by="clust")

    out_path = output_dir / gene[0] / f"{gene}_nmfinfo_final.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_pos.to_csv(out_path)
    return out_path
