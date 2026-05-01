# Import necessary packages
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact

# Config
root_dir = "../../3_drug_ml/output/MLP_outputs"   
nmf_root = "../../1_nmf/output"                   
mapping_csv = "../data/uniprot_gene_map.csv"   
output_root = "../output/logodds_results"
models = ["mlp"]

os.makedirs(output_root, exist_ok=True)


# Read UniProt ID from command line:
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py UNIPROT_ID")
    sys.exit(1)

uniprot_id = sys.argv[1].upper().strip()

# Functions
def run_fisher(df: pd.DataFrame) -> pd.DataFrame:
    df_collapsed = (
        df.groupby(["protein", "res", "cluster_id", "drug"], as_index=False)
          .agg({"prediction": "mean"})
    )
    df_collapsed["prediction"] = (df_collapsed["prediction"] >= 0.90).astype(int)

    results = []
    for (protein, drug, cluster_id), sub in df_collapsed.groupby(["protein", "drug", "cluster_id"]):
        all_sub = df_collapsed[
            (df_collapsed["protein"] == protein) &
            (df_collapsed["drug"] == drug)
        ]
        in_cluster = sub
        out_cluster = all_sub[all_sub["cluster_id"] != cluster_id]

        if len(out_cluster) == 0:
            continue

        a = int(in_cluster["prediction"].sum())
        b = int(len(in_cluster) - a)
        c = int(out_cluster["prediction"].sum())
        d = int(len(out_cluster) - c)

        try:
            oddsratio, _ = fisher_exact([[a, b], [c, d]])
        except ValueError:
            oddsratio = np.nan

        if np.isfinite(oddsratio) and oddsratio > 0:
            log_odds = np.log2(oddsratio)
        elif np.isinf(oddsratio):
            log_odds = np.sign(oddsratio) * 6
        else:
            log_odds = np.nan

        results.append({
            "cluster_id": cluster_id,
            "log2_odds_ratio": log_odds
        })

    return pd.DataFrame(results)


def find_drug_dirs(root: str, model_list: list[str]) -> list[tuple[str, str]]:
    drug_dirs = []
    for current_root, dirs, files in os.walk(root):
        if any(os.path.exists(os.path.join(current_root, f"{m}_predictions.csv")) for m in model_list):
            drug_name = os.path.basename(current_root.rstrip(os.sep))
            drug_dirs.append((current_root, drug_name))
    return drug_dirs


def load_uniprot_to_gene_map(mapping_csv_path: str) -> dict:
    map_df = pd.read_csv(mapping_csv_path)
    map_df.columns = [c.strip().lower() for c in map_df.columns]

    if "uniprot_id" not in map_df.columns:
        raise ValueError(f"{mapping_csv_path} must contain a 'uniprot_id' column")
    if "gene" not in map_df.columns:
        raise ValueError(f"{mapping_csv_path} must contain a 'gene' column")

    map_df["uniprot_id"] = map_df["uniprot_id"].astype(str).str.strip()
    map_df["gene"] = map_df["gene"].astype(str).str.strip()

    map_df = map_df.dropna(subset=["uniprot_id", "gene"])
    map_df = map_df[(map_df["uniprot_id"] != "") & (map_df["gene"] != "")]

    return dict(zip(map_df["uniprot_id"], map_df["gene"]))


# Load mapping
uniprot_to_gene = load_uniprot_to_gene_map(mapping_csv)

# Validate protein exists
protein_dirs = sorted([
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
])

if uniprot_id not in protein_dirs:
    print(f"Error: '{uniprot_id}' not found in {root_dir}")
    print(f"Available folders: {protein_dirs}")
    sys.exit(1)

proteins_to_run = [uniprot_id]

# Main loop
for uniprot_id in proteins_to_run:
    print(f"\n=== Processing UniProt: {uniprot_id} ===")

    if uniprot_id not in uniprot_to_gene:
        print(f"Skipping {uniprot_id}: no mapping found in {mapping_csv}")
        continue

    gene_name = uniprot_to_gene[uniprot_id]
    nmf_letter = gene_name[0].upper()
    cluster_path = os.path.join(nmf_root, nmf_letter, f"{gene_name}_nmfinfo_final.csv")

    if not os.path.exists(cluster_path):
        print(f"Skipping {uniprot_id}: no cluster file found at {cluster_path}")
        continue

    print(f"Mapped {uniprot_id} -> {gene_name}")

    cluster_df = pd.read_csv(cluster_path).rename(
        columns={"res": "residue_position", "clust": "cluster_id"}
    )
    cluster_df["residue_position"] = cluster_df["residue_position"].astype(int)
    cluster_df["cluster_id"] = cluster_df["cluster_id"].astype(int)

    base_dir = os.path.join(root_dir, uniprot_id)
    drug_dirs = find_drug_dirs(base_dir, models)

    if not drug_dirs:
        print(f"No drug folders found under {base_dir}")
        continue

    output_dir = os.path.join(output_root, gene_name)
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    for drug_dir, drug in drug_dirs:
        print(f"  -> {drug}")
        summary = pd.DataFrame()

        for model in models:
            fpath = os.path.join(drug_dir, f"{model}_predictions.csv")
            if not os.path.exists(fpath):
                continue

            df = pd.read_csv(fpath)

            if "residue_position" not in df.columns:
                for alt in ["res", "res_pos", "position"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "residue_position"})
                        break

            if "residue_position" not in df.columns:
                raise ValueError(f"{fpath} missing residue_position column.")

            df["residue_position"] = df["residue_position"].astype(int)

            if "y_pred_binary" in df.columns:
                df["prediction"] = df["y_pred_binary"].astype(int)
            elif "observed_label" in df.columns:
                df["prediction"] = df["observed_label"].astype(int)
            else:
                raise ValueError(f"{fpath} missing prediction column.")

            df = df.merge(
                cluster_df[["residue_position", "cluster_id"]],
                how="inner",
                on="residue_position"
            )

            df["protein"] = gene_name
            df["res"] = df["residue_position"]
            df["drug"] = drug

            res_df = run_fisher(df).rename(columns={"log2_odds_ratio": model})
            summary = res_df if summary.empty else pd.merge(summary, res_df, on="cluster_id", how="outer")

        if not summary.empty:
            out_path = os.path.join(output_dir, f"{gene_name}_{drug}_logodds.csv")
            summary.to_csv(out_path, index=False)
            all_results.append(summary.assign(drug=drug))
            print(f"Saved: {out_path}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        melted = combined.melt(
            id_vars=["drug", "cluster_id"],
            value_vars=models,
            var_name="model",
            value_name="log2_odds_ratio"
        )

        summary = (
            melted.groupby(["drug", "model"])["log2_odds_ratio"]
            .max()
            .reset_index()
        )

        for model in models:
            sub = summary[summary["model"] == model].sort_values(
                "log2_odds_ratio",
                ascending=False
            )
            out_path = os.path.join(output_dir, f"sorted_{model}.csv")
            sub.to_csv(out_path, index=False)
            print(f"Saved: {out_path}")

print("Done!")