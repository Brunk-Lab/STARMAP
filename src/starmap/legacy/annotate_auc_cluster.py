# Import necessary packages
from pathlib import Path
import sys
import re
import pandas as pd
import numpy as np

# Config
ccle_csv = "../data/CCLE_cell_line_drugsensitivity_10072023.csv"
dist_dir = Path("../output/dist_files")
map_csv = Path("../data/gene_uniprot_res_clust.csv")

id_col_dist = "ID"
sens_col_name = "sensitivity"
parquet_suffix = ".parquet"

# Read UniProt ID from command line:
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py UNIPROT_ID")
    sys.exit(1)
    
target_uid = sys.argv[1].upper().strip()

# regex for residue extraction
re_pos = re.compile(r"^p\.(?:[A-Z\*]|[A-Z][a-z]{2})(\d+)", re.ASCII)

# Functions
def per_drug_p25(df_ccle: pd.DataFrame) -> dict:
    return (
        df_ccle.dropna(subset=["drug", "AUC_CTRP"])
        .groupby("drug")["AUC_CTRP"]
        .quantile(0.25)
        .astype(float)
        .to_dict()
    )


def extract_pos(s):
    if not isinstance(s, str):
        return None
    m = re_pos.match(s.strip())
    return int(m.group(1)) if m else None


def load_mapping(map_csv: Path) -> dict:
    df = pd.read_csv(
        map_csv,
        dtype={"uniprot_ID": str, "gene": str, "clust": "Int64", "res": "Int64"}
    )
    df = df.dropna(subset=["uniprot_ID", "res", "clust"])
    df["uniprot_ID"] = df["uniprot_ID"].astype(str)
    df["res"] = df["res"].astype(int)
    df["clust"] = df["clust"].astype(int)

    dups = df.duplicated(subset=["uniprot_ID", "res"], keep=False)
    if dups.any():
        print(f"[warn] {int(dups.sum())} duplicated mapping rows → keeping first")
        df = df.drop_duplicates(subset=["uniprot_ID", "res"], keep="first")

    return {(r.uniprot_ID, r.res): r.clust for r in df.itertuples(index=False)}


def process_protein(uid, ccle, p25_map, key_to_clust):
    prot_dir = dist_dir / uid
    if not prot_dir.exists():
        print(f"[skip] missing folder: {uid}")
        return

    files = list(prot_dir.rglob(f"{uid}_Distances.csv"))
    if not files:
        print(f"[skip] no distance file for {uid}")
        return

    for f in files:
        print(f"[read] {f}")

        dist = pd.read_csv(f)
        if id_col_dist not in dist.columns:
            print(f"[warn] missing {id_col_dist} → skipped")
            continue

        # -------------------------
        # merge CCLE
        # -------------------------
        merged = dist.merge(ccle, how="left", left_on=id_col_dist, right_on="cell_line")
        merged.drop(columns=["cell_line"], errors="ignore", inplace=True)

        merged["p25"] = merged["drug"].map(p25_map)
        merged[sens_col_name] = ""

        m = (
            merged["AUC_CTRP"].notna()
            & merged["p25"].notna()
            & merged["drug"].notna()
        )

        merged.loc[m, sens_col_name] = np.where(
            merged.loc[m, "AUC_CTRP"] <= merged.loc[m, "p25"],
            "sensitive",
            "not sensitive",
        )

        merged.drop(columns=["p25"], inplace=True)

        # write CSV
        labeled_csv = f.with_name(f"{uid}_Distances_Labeled.csv")
        merged.to_csv(labeled_csv, index=False)

        print(f"[ok] wrote csv → {labeled_csv.name}")

        # Add cluster annotations
        if "protein_change" not in merged.columns:
            print(f"[warn] no protein_change column → skipping clust")
            continue

        df2 = merged.dropna(subset=["AUC_CTRP"]).copy()

        pos = df2["protein_change"].map(extract_pos).astype("Int64")
        keys = list(zip([uid] * len(pos), pos.fillna(-1).astype(int)))
        df2["clust"] = [key_to_clust.get(k) if k[1] != -1 else None for k in keys]

        # write parquet
        out_parquet = labeled_csv.with_suffix(parquet_suffix)

        try:
            try:
                df2.to_parquet(out_parquet, index=False, engine="pyarrow")
            except Exception:
                df2.to_parquet(out_parquet, index=False, engine="fastparquet")
        except Exception as e:
            print(f"[error] parquet failed: {e}")
            continue

        n = len(df2)
        n_unmatched = int(pd.isna(df2["clust"]).sum())

        print(f"[ok] wrote parquet → {out_parquet.name}")
        if n:
            print(f"[stats] {uid} coverage: {(n - n_unmatched) * 100.0 / n:.2f}%")

# Main
def main():
    print("[read] ccle")
    ccle = pd.read_csv(ccle_csv, usecols=["cell_line", "drug", "AUC_CTRP"])
    ccle["AUC_CTRP"] = pd.to_numeric(ccle["AUC_CTRP"], errors="coerce")
    ccle = ccle.groupby(["cell_line", "drug"], as_index=False)["AUC_CTRP"].mean()

    p25_map = per_drug_p25(ccle)
    print(f"[info] computed p25 for {len(p25_map)} drugs")

    print("[read] mapping")
    key_to_clust = load_mapping(map_csv)

    # decide which proteins to run
    if target_uid:
        uids = [target_uid]
    else:
        uids = [p.name for p in dist_dir.iterdir() if p.is_dir()]
        print(f"[info] running all proteins: {len(uids)} found")

    for uid in uids:
        print(f"\n=== processing {uid} ===")
        process_protein(uid, ccle, p25_map, key_to_clust)


if __name__ == "__main__":
    main()