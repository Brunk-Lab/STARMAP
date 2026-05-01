from __future__ import annotations

import glob
import gzip
import json
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from esda.getisord import G_Local
from libpysal.weights import Queen
from shapely.geometry import Point, mapping


def turn_to_map(df: pd.DataFrame, colname_list: list[str]) -> dict:
    features = []
    for _, row in df.iterrows():
        features.append({
            "type": "Feature",
            "geometry": mapping(Point(row["x_axis"], row["y_axis"])),
            "properties": {col: row[col] for col in colname_list},
        })
    return {"type": "FeatureCollection", "features": features}


def calculate_gi_statistics(df_pos: pd.DataFrame, colname_list: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    geojson = turn_to_map(df_pos, colname_list)
    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    for gsea in colname_list:
        weights = Queen.from_dataframe(gdf, use_index=False)
        values = gdf[gsea].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g = G_Local(values, weights, n_jobs=1)
        gdf[f"Gi_{gsea}"] = g.Zs
    gdf["Gi_sum"] = gdf[[f"Gi_{gsea}" for gsea in colname_list]].sum(axis=1)
    cluster = df_pos[["res", "x_axis", "y_axis", "altitude", "clust"]].reset_index(drop=True).copy()
    cluster["Gi_sum"] = gdf["Gi_sum"].values
    score_df = pd.DataFrame({
        "scores": [cluster.groupby("clust")["Gi_sum"].mean().tolist()],
        "counts": [cluster.groupby("clust")["Gi_sum"].apply(lambda x: (x > 0).sum()).tolist()],
    })
    return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore")), score_df


def prepare_gene_pathway_scores(gene: str, pathway_path: str | Path, position_cellline_csv: str | Path, nmf_root: str | Path):
    gsea = pd.read_csv(pathway_path).dropna().set_index("Unnamed: 0")
    pos_df = pd.read_csv(position_cellline_csv)
    pos_df = pos_df[pos_df["gene"].astype(str).str.upper() == gene.upper()].reset_index(drop=True)
    pos_df["gsea_score"] = pos_df["Tumor_Sample_Barcode"].apply(lambda x: gsea.loc[x].iloc[0] if x in gsea.index else 0)
    pos_df = pos_df[["position", "gsea_score"]].set_index("position").query("gsea_score != 0")
    nmf_path = Path(nmf_root) / gene[0].upper() / f"{gene}_nmfinfo_final.csv"
    df_pos = pd.read_csv(nmf_path)
    if "res" not in df_pos.columns:
        df_pos = df_pos.rename(columns={df_pos.columns[0]: "res"})
    df_pos["res"] = df_pos["res"].astype(int)
    mapped = []
    for atom in df_pos["res"]:
        val = pos_df.loc[atom, "gsea_score"] if atom in pos_df.index else 0
        mapped.append([val] if not isinstance(val, pd.Series) else val.tolist())
    df_pos["gsea_string"] = [",".join(map(str, x)) for x in mapped]
    max_len = max(df_pos["gsea_string"].apply(lambda x: len(x.split(","))))
    colnames = [f"gsea{i}" for i in range(max_len)]
    split_df = df_pos["gsea_string"].str.split(",", expand=True)
    split_df.columns = colnames
    for col in colnames:
        df_pos[col] = split_df[col].fillna("0").astype(float)
    return df_pos, colnames


def run_gistar_for_gene(
    gene: str,
    gsea_dir: str | Path,
    position_cellline_csv: str | Path,
    nmf_root: str | Path,
    output_root: str | Path,
    gzip_gdf: bool = True,
) -> Path | None:
    gene = gene.upper().strip()
    paths = sorted(glob.glob(str(Path(gsea_dir) / "*_GSEA.csv")))
    if not paths:
        raise FileNotFoundError(f"No *_GSEA.csv files found in {gsea_dir}")
    map_dir = Path(output_root) / "3D_files" / gene[0]
    score_dir = Path(output_root) / "scores" / gene[0]
    map_dir.mkdir(parents=True, exist_ok=True); score_dir.mkdir(parents=True, exist_ok=True)
    all_scores = []
    for pathway_path in paths:
        trn = Path(pathway_path).name.replace("_GSEA.csv", "")
        try:
            df_pos, colnames = prepare_gene_pathway_scores(gene, pathway_path, position_cellline_csv, nmf_root)
            gdf, score_df = calculate_gi_statistics(df_pos, colnames)
            score_df["gene"] = gene; score_df["trn"] = trn; score_df["pathway_file"] = Path(pathway_path).name
            all_scores.append(score_df)
            gdf_out = map_dir / f"{gene}_{trn}_gdf.csv"
            gdf.to_csv(gdf_out, index=False)
            if gzip_gdf:
                with open(gdf_out, "rb") as src, gzip.open(str(gdf_out) + ".gz", "wb", compresslevel=9) as dst:
                    dst.write(src.read())
                gdf_out.unlink()
        except Exception as exc:
            print(f"[WARN] failed on {pathway_path}: {exc}")
    if not all_scores:
        return None
    final = pd.concat(all_scores, ignore_index=True)
    out = score_dir / f"{gene}_scores_all_trns.csv"
    final.to_csv(out, index=False)
    return out
