from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from scipy.stats import ranksums


def normalize_drug_name(name: object) -> str:
    name = str(name).lower().strip()
    name = re.sub(r"^(random|positive|negative)_", "", name)
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[^a-z0-9\-]+", "", name)
    return name


def _mean_first_numeric_cols(df: pd.DataFrame, n: int = 10) -> pd.Series:
    cols = [c for c in df.columns[:n] if c != "Unnamed: 0"]
    return df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def run_tahoe_confidence(
    gene: str,
    sorted_logodds_csv: str | Path,
    perdrug_dir: str | Path,
    output_dir: str | Path,
    pvalue_cutoff: float = 0.05,
) -> Path:
    gene = gene.upper().strip()
    logodds = pd.read_csv(sorted_logodds_csv).reset_index()
    logodds["index"] = logodds["index"] + 1
    control = pd.read_csv(Path(perdrug_dir) / "ssgsea_cluster_all_trns_DMSO_TF.csv")
    control["mean_expr"] = _mean_first_numeric_cols(control)
    files = [f for f in os.listdir(perdrug_dir) if f.endswith(".csv")]
    drugs = sorted(set(f.replace("ssgsea_cluster_all_trns_", "").replace("ssgsea_other_all_trns_", "").replace(".csv", "") for f in files))
    rows = []
    for drug in drugs:
        clust_path = Path(perdrug_dir) / f"ssgsea_cluster_all_trns_{drug}.csv"
        other_path = Path(perdrug_dir) / f"ssgsea_other_all_trns_{drug}.csv"
        if not clust_path.exists() or not other_path.exists():
            continue
        clust = pd.read_csv(clust_path); other = pd.read_csv(other_path)
        clust["mean_expr"] = _mean_first_numeric_cols(clust)
        other["mean_expr"] = _mean_first_numeric_cols(other)
        rows.append({
            "Drug": drug,
            "clust_testvcontrol": ranksums(clust["mean_expr"], control["mean_expr"]).pvalue,
            "test_clustvother": ranksums(clust["mean_expr"], other["mean_expr"]).pvalue,
        })
    stat = pd.DataFrame(rows)
    stat["norm_drug"] = stat["Drug"].apply(normalize_drug_name)
    logodds["norm_drug"] = logodds["drug"].apply(normalize_drug_name)
    merged = pd.merge(logodds, stat, on="norm_drug", how="inner").drop_duplicates().reset_index(drop=True)
    original_rank = merged["index"].copy()
    merged = merged.sort_values(["log2_odds_ratio", "clust_testvcontrol"], ascending=[False, True]).reset_index(drop=True)
    merged["index"] = original_rank.values[: len(merged)]
    merged["rank"] = merged.index + 1
    sub = merged[["index", "rank", "clust_testvcontrol", "test_clustvother"]].copy().sort_values("rank")
    sub["significant"] = sub["clust_testvcontrol"] < pvalue_cutoff
    sub["cum_significant"] = sub["significant"].cumsum()
    sub["confidence"] = sub["cum_significant"] / sub["rank"]
    emp = sub[["index", "confidence"]].copy()
    denom = emp["confidence"].max() - emp["confidence"].min()
    emp["norm_confidence"] = 0.0 if denom == 0 else (emp["confidence"] - emp["confidence"].min()) / denom
    emp["protein"] = gene
    out = Path(output_dir) / f"{gene}_emp.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    emp.to_csv(out, index=False)
    return out
