from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def run_fisher(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    df_collapsed = (
        df.groupby(["protein", "res", "cluster_id", "drug"], as_index=False)
        .agg({"prediction": "mean"})
    )
    df_collapsed["prediction"] = (df_collapsed["prediction"] >= threshold).astype(int)

    results = []
    for (protein, drug, cluster_id), sub in df_collapsed.groupby(["protein", "drug", "cluster_id"]):
        all_sub = df_collapsed[(df_collapsed["protein"] == protein) & (df_collapsed["drug"] == drug)]
        out_cluster = all_sub[all_sub["cluster_id"] != cluster_id]
        if len(out_cluster) == 0:
            continue

        a = int(sub["prediction"].sum())
        b = int(len(sub) - a)
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

        results.append({"cluster_id": cluster_id, "log2_odds_ratio": log_odds})
    return pd.DataFrame(results)


def find_drug_dirs(root: str | Path, model_list: list[str]) -> list[tuple[Path, str]]:
    drug_dirs = []
    for current_root, _, _ in os.walk(root):
        current_root = Path(current_root)
        if any((current_root / f"{model}_predictions.csv").exists() for model in model_list):
            drug_dirs.append((current_root, current_root.name))
    return drug_dirs


def load_uniprot_to_gene_map(mapping_csv_path: str | Path) -> dict[str, str]:
    map_df = pd.read_csv(mapping_csv_path)
    map_df.columns = [c.strip().lower() for c in map_df.columns]
    if "uniprot_id" not in map_df.columns or "gene" not in map_df.columns:
        raise ValueError("Mapping CSV must contain 'uniprot_id' and 'gene' columns")
    map_df["uniprot_id"] = map_df["uniprot_id"].astype(str).str.strip()
    map_df["gene"] = map_df["gene"].astype(str).str.strip()
    map_df = map_df.dropna(subset=["uniprot_id", "gene"])
    return dict(zip(map_df["uniprot_id"], map_df["gene"]))


def run_logodds_for_uniprot(
    uniprot_id: str,
    mlp_root: str | Path,
    nmf_root: str | Path,
    mapping_csv: str | Path,
    output_root: str | Path,
    models: list[str] | None = None,
) -> Path | None:
    models = models or ["mlp"]
    uniprot_id = uniprot_id.upper().strip()
    mlp_root = Path(mlp_root)
    nmf_root = Path(nmf_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    uniprot_to_gene = load_uniprot_to_gene_map(mapping_csv)
    if uniprot_id not in uniprot_to_gene:
        raise KeyError(f"No mapping found for {uniprot_id}")

    gene_name = uniprot_to_gene[uniprot_id]
    cluster_path = nmf_root / gene_name[0].upper() / f"{gene_name}_nmfinfo_final.csv"
    if not cluster_path.exists():
        raise FileNotFoundError(f"Missing cluster file: {cluster_path}")

    cluster_df = pd.read_csv(cluster_path).rename(columns={"res": "residue_position", "clust": "cluster_id"})
    cluster_df["residue_position"] = cluster_df["residue_position"].astype(int)
    cluster_df["cluster_id"] = cluster_df["cluster_id"].astype(int)

    base_dir = mlp_root / uniprot_id
    drug_dirs = find_drug_dirs(base_dir, models)
    output_dir = output_root / gene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for drug_dir, drug in drug_dirs:
        summary = pd.DataFrame()
        for model in models:
            fpath = drug_dir / f"{model}_predictions.csv"
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            if "residue_position" not in df.columns:
                for alt in ["res", "res_pos", "position"]:
                    if alt in df.columns:
                        df = df.rename(columns={alt: "residue_position"})
                        break
            if "residue_position" not in df.columns:
                raise ValueError(f"{fpath} missing residue_position column")
            df["residue_position"] = df["residue_position"].astype(int)
            if "y_pred_binary" in df.columns:
                df["prediction"] = df["y_pred_binary"].astype(int)
            elif "observed_label" in df.columns:
                df["prediction"] = df["observed_label"].astype(int)
            else:
                raise ValueError(f"{fpath} missing y_pred_binary or observed_label column")
            df = df.merge(cluster_df[["residue_position", "cluster_id"]], how="inner", on="residue_position")
            df["protein"] = gene_name
            df["res"] = df["residue_position"]
            df["drug"] = drug
            res_df = run_fisher(df).rename(columns={"log2_odds_ratio": model})
            summary = res_df if summary.empty else pd.merge(summary, res_df, on="cluster_id", how="outer")
        if not summary.empty:
            summary.to_csv(output_dir / f"{gene_name}_{drug}_logodds.csv", index=False)
            all_results.append(summary.assign(drug=drug))

    if not all_results:
        return None

    combined = pd.concat(all_results, ignore_index=True)
    melted = combined.melt(id_vars=["drug", "cluster_id"], value_vars=models, var_name="model", value_name="log2_odds_ratio")
    summary = melted.groupby(["drug", "model"])["log2_odds_ratio"].max().reset_index()
    last_path = None
    for model in models:
        sub = summary[summary["model"] == model].sort_values("log2_odds_ratio", ascending=False)
        last_path = output_dir / f"sorted_{model}.csv"
        sub.to_csv(last_path, index=False)
    return last_path
