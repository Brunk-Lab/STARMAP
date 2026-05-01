#!/usr/bin/env python3
import os
import sys
import json
import glob
import warnings
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, mapping
from libpysal.weights import Queen
from esda.getisord import G_Local


# Functions
def turn_to_map(df, colname_list):
    """Convert residue coordinates and GSEA values into a GeoJSON FeatureCollection."""
    features = []
    for _, row in df.iterrows():
        props = {col: row[col] for col in colname_list}
        point = Point(row["x_axis"], row["y_axis"])
        feature = {
            "type": "Feature",
            "geometry": mapping(point),
            "properties": props
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}


def calculate_gi_statistics(colname_list, cluster, geojson_file):
    """Compute local Getis-Ord Gi* Z-scores for each GSEA column."""
    gdf = gpd.read_file(geojson_file)

    for gsea in colname_list:
        weights = Queen.from_dataframe(gdf, use_index=False)
        values = gdf[gsea].values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            g = G_Local(values, weights, n_jobs=1)
        gdf[f"Gi_{gsea}"] = g.Zs

    gdf["Gi_sum"] = gdf[[f"Gi_{gsea}" for gsea in colname_list]].sum(axis=1)

    cluster = cluster.reset_index(drop=True).copy()
    cluster["Gi_sum"] = gdf["Gi_sum"].iloc[cluster["res"] - 1].values

    cluster_scores = cluster.groupby("clust")["Gi_sum"].mean().tolist()
    cluster_counts = cluster.groupby("clust")["Gi_sum"].apply(lambda x: (x > 0).sum()).tolist()

    score_df = pd.DataFrame({
        "scores": [cluster_scores],
        "counts": [cluster_counts]
    })
    return gdf, score_df


def process_gene_pathway(gene, pathway_path):
    """Load one gene and one pathway file, align GSEA scores with residues."""
    pathway_file = os.path.basename(pathway_path)

    gsea = pd.read_csv(pathway_path).dropna().set_index("Unnamed: 0")

    df = pd.read_csv("../data/ccle_gene_position_cellline.csv")
    df = df[df["gene"] == gene].reset_index(drop=True)

    df["gsea_score"] = df["Tumor_Sample_Barcode"].apply(
        lambda x: gsea.loc[x].iloc[0] if x in gsea.index else 0
    )
    df = df[["position", "gsea_score"]].set_index("position").query("gsea_score != 0")

    nmf_path = f"../../1_nmf/output/{gene[0]}/{gene}_nmfinfo_final.csv"
    df_pos = pd.read_csv(nmf_path).reset_index()
    df_pos["res"] = df_pos["res"].astype(int)
    cluster = df_pos[["res", "x_axis", "y_axis", "altitude", "clust"]].copy()

    mapped_scores = []
    for atom in df_pos["res"]:
        val = df.loc[atom, "gsea_score"] if atom in df.index else 0
        mapped_scores.append([val] if not isinstance(val, pd.Series) else val.tolist())

    df_pos["gsea"] = mapped_scores
    df_pos["gsea_string"] = df_pos["gsea"].apply(lambda x: ",".join(map(str, x)))

    max_len = max(df_pos["gsea_string"].apply(lambda x: len(x.split(","))))
    colnames = [f"gsea{i}" for i in range(max_len)]

    split_df = df_pos["gsea_string"].str.split(",", expand=True)
    split_df.columns = colnames

    for col in colnames:
        df_pos[col] = split_df[col].fillna("0").astype(float)

    return df_pos, colnames, cluster, pathway_file


# Run Gi* method
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py GENE")
    sys.exit(1)

gene = sys.argv[1].upper().strip()

gsea_dir = "../data/GSEA_files"
pathway_paths = sorted(glob.glob(os.path.join(gsea_dir, "*_GSEA.csv")))

if not pathway_paths:
    raise FileNotFoundError(f"No *_GSEA.csv files found in {gsea_dir}")

print(f"Running GI* for gene: {gene}")
print(f"Found {len(pathway_paths)} pathway files")

out_dir = f"../output/3D_files/{gene[0]}"
score_dir = f"../output/scores/{gene[0]}"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(score_dir, exist_ok=True)

all_score_dfs = []

# Loop across all TRNs
for pathway_path in pathway_paths:
    pathway_file = os.path.basename(pathway_path)
    trn = pathway_file.replace("_GSEA.csv", "")

    print(f"Processing {pathway_file}...")

    try:
        df_pos, colnames, cluster, pathway_file = process_gene_pathway(gene, pathway_path)

        geojson_path = f"{out_dir}/{gene}_{trn}_map.geojson"
        with open(geojson_path, "w") as f:
            json.dump(turn_to_map(df_pos, colnames), f)

        gdf, score_df = calculate_gi_statistics(colnames, cluster, geojson_path)

        score_df["gene"] = gene
        score_df["trn"] = trn
        score_df["pathway_file"] = pathway_file
        all_score_dfs.append(score_df)

        gdf_out = f"{out_dir}/{gene}_{trn}_gdf.csv"
        gdf.to_csv(gdf_out, index=False)
        os.system(f"gzip -9 '{gdf_out}'")

        if os.path.exists(geojson_path):
            os.remove(geojson_path)

        print(f"{gene} done for {trn}")

    except Exception as e:
        print(f"Failed on {pathway_file}: {e}")

# Save final scores df
if all_score_dfs:
    final_scores = pd.concat(all_score_dfs, ignore_index=True)
    out_path = f"{score_dir}/{gene}_scores_all_trns.csv"
    final_scores.to_csv(out_path, index=False)
    print(f"Saved all scores to {out_path}")
else:
    print("No score files were generated.")
