from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

RE_POS = re.compile(r"^p\.(?:[A-Z\*]|[A-Z][a-z]{2})(\d+)", re.ASCII)


def per_drug_quantile(df_ccle: pd.DataFrame, quantile: float = 0.25) -> dict:
    return (
        df_ccle.dropna(subset=["drug", "AUC_CTRP"])
        .groupby("drug")["AUC_CTRP"]
        .quantile(quantile)
        .astype(float)
        .to_dict()
    )


def extract_pos(value):
    if not isinstance(value, str):
        return None
    m = RE_POS.match(value.strip())
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", value)
    return int(m.group(1)) if m else None


def load_cluster_mapping(map_csv: str | Path) -> dict[tuple[str, int], int]:
    df = pd.read_csv(map_csv)
    cols = {c.lower(): c for c in df.columns}
    uid_col = cols.get("uniprot_id") or cols.get("uniprot_id") or cols.get("uniprot_id") or cols.get("uniprot_id") or cols.get("uniprot_id")
    uid_col = uid_col or cols.get("uniprot_id") or cols.get("uniprot_id")
    # accept the original mixed-case column
    uid_col = uid_col or ("uniprot_ID" if "uniprot_ID" in df.columns else None)
    res_col = cols.get("res") or cols.get("residue") or cols.get("residue_position")
    clust_col = cols.get("clust") or cols.get("cluster") or cols.get("cluster_id")
    if uid_col is None or res_col is None or clust_col is None:
        raise ValueError("Mapping CSV must include UniProt, residue, and cluster columns")
    out = df.dropna(subset=[uid_col, res_col, clust_col]).copy()
    out[uid_col] = out[uid_col].astype(str).str.strip().str.upper()
    out[res_col] = out[res_col].astype(int)
    out[clust_col] = out[clust_col].astype(int)
    out = out.drop_duplicates(subset=[uid_col, res_col], keep="first")
    return {(r[uid_col], int(r[res_col])): int(r[clust_col]) for _, r in out.iterrows()}


def label_distance_table(
    distance_df: pd.DataFrame,
    ccle_df: pd.DataFrame,
    uniprot_id: str,
    cluster_map: dict[tuple[str, int], int] | None = None,
    sensitivity_quantile: float = 0.25,
    id_col_dist: str = "ID",
) -> pd.DataFrame:
    ccle = ccle_df[["cell_line", "drug", "AUC_CTRP"]].copy()
    ccle["AUC_CTRP"] = pd.to_numeric(ccle["AUC_CTRP"], errors="coerce")
    ccle = ccle.groupby(["cell_line", "drug"], as_index=False)["AUC_CTRP"].mean()
    pmap = per_drug_quantile(ccle, sensitivity_quantile)
    merged = distance_df.merge(ccle, how="left", left_on=id_col_dist, right_on="cell_line")
    merged.drop(columns=["cell_line"], errors="ignore", inplace=True)
    merged["p_threshold"] = merged["drug"].map(pmap)
    mask = merged["AUC_CTRP"].notna() & merged["p_threshold"].notna() & merged["drug"].notna()
    merged["sensitivity"] = np.nan
    merged.loc[mask, "sensitivity"] = np.where(
        merged.loc[mask, "AUC_CTRP"] <= merged.loc[mask, "p_threshold"],
        "sensitive",
        "not sensitive",
    )
    merged.drop(columns=["p_threshold"], inplace=True)
    if cluster_map is not None and "protein_change" in merged.columns:
        pos = merged["protein_change"].map(extract_pos)
        uid = uniprot_id.upper().strip()
        merged["clust"] = [cluster_map.get((uid, int(p))) if pd.notna(p) else None for p in pos]
    return merged


def run_annotate_auc_cluster(
    uniprot_id: str,
    ccle_csv: str | Path,
    dist_dir: str | Path,
    map_csv: str | Path | None,
    output_suffix: str = "_Distances_Labeled.parquet",
    sensitivity_quantile: float = 0.25,
) -> Path:
    uid = uniprot_id.upper().strip()
    dist_path = Path(dist_dir) / uid / f"{uid}_Distances.csv"
    if not dist_path.exists():
        raise FileNotFoundError(dist_path)
    dist = pd.read_csv(dist_path)
    ccle = pd.read_csv(ccle_csv, usecols=["cell_line", "drug", "AUC_CTRP"])
    cmap = load_cluster_mapping(map_csv) if map_csv else None
    labeled = label_distance_table(dist, ccle, uid, cmap, sensitivity_quantile=sensitivity_quantile)
    out = dist_path.with_name(f"{uid}{output_suffix}")
    if out.suffix == ".parquet":
        labeled.dropna(subset=["AUC_CTRP"], how="any").to_parquet(out, index=False)
    else:
        labeled.to_csv(out, index=False)
    return out
