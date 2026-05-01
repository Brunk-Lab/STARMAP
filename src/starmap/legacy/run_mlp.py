# Import necessary packages
import os
import re
import gc
import uuid
import warnings
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    f1_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping

# Config
warnings.filterwarnings("ignore", category=UserWarning)

os.environ.update({
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

torch.manual_seed(42)
np.random.seed(42)

root = Path("../")
dist_root = root / "output/dist_files"
mlp_output_root = root / "output/MLP_outputs"
mlp_output_root.mkdir(parents=True, exist_ok=True)

test_size = 0.30
random_state = 42
min_samples_total = 20
min_samples_per_class = 5
min_groups = 4
use_clust = True

# Read UniProt ID from command line:
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py UNIPROT_ID")
    sys.exit(1)
    
target_uid = sys.argv[1].upper().strip()
target_drug = None  # set to a drug string if you want only one drug


# Column patterns
pat_dtas = re.compile(r"^DTAS_(\d+)$")
pat_dtbs = re.compile(r"^DTBS_(\d+)$")
pat_asxyz = re.compile(r"^AS_(\d+)_(x|y|z)$")
pat_bsxyz = re.compile(r"^BS_(\d+)_(x|y|z)$")
pat_dtasr = re.compile(r"^DTASR_(\d+)-(\d+)$")
pat_dtbsr = re.compile(r"^DTBSR_(\d+)-(\d+)$")
pat_asrxyz = re.compile(r"^ASR_(\d+)-(\d+)_(x|y|z)$")
pat_bsrxyz = re.compile(r"^BSR_(\d+)-(\d+)_(x|y|z)$")
pat_caxyz = re.compile(r"^Mut_CA_(x|y|z)$")

exclude_cols = {
    "ID", "CELL_LINE", "SIFT", "LIKELY_LOF", "protein_change",
    "drug", "AUC_CTRP", "sensitivity", "clust"
}

# Functions
def sanitize_drug_id(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", str(s)).strip("_")


def ensure_clust_dummies(df: pd.DataFrame) -> pd.DataFrame:
    if not use_clust or "clust" not in df.columns:
        return df

    cl = df["clust"]
    if pd.api.types.is_integer_dtype(cl) or pd.api.types.is_float_dtype(cl):
        cl_str = cl.astype("Int64").astype(str).radd("c")
    else:
        cl_str = cl.astype(str).apply(
            lambda s: f"c{s}" if s and str(s).lower() != "nan" else "c_missing"
        )

    cl_str = cl_str.fillna("c_missing").replace({
        "nan": "c_missing",
        "c<NA>": "c_missing"
    })

    dummies = pd.get_dummies(cl_str, prefix="clust", dtype=float)
    return pd.concat([df, dummies], axis=1)


def pick_feature_columns(df: pd.DataFrame):
    cols = []
    for c in df.columns:
        if c in exclude_cols:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith("clust_"):
            cols.append(c)
            continue
        if any(p.match(c) for p in [pat_dtas, pat_dtbs, pat_asxyz, pat_bsxyz, pat_caxyz]):
            cols.append(c)
            continue
        if any(p.match(c) for p in [pat_dtasr, pat_dtbsr, pat_asrxyz, pat_bsrxyz]):
            cols.append(c)
            continue
    return sorted(cols)


def _normalize_labels(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {
        "sensitive": 1.0,
        "not sensitive": 0.0,
    }
    bad = s[~s.isin(mapping.keys())].unique().tolist()
    if bad and bad != [""]:
        raise SystemExit(
            f"Unexpected sensitivity values: {bad} | Expected: {list(mapping.keys())}"
        )
    mapped = s.map(mapping)
    return mapped.dropna().astype(float)


def prepare_xy_groups(df: pd.DataFrame, feat_cols):
    y = _normalize_labels(df["sensitivity"])
    x = df.loc[:, feat_cols].copy()
    groups = df.get(
        "CELL_LINE",
        pd.Series(["_NA_"] * len(df), index=df.index)
    ).astype(str)

    mask = x.notna().all(axis=1) & y.isin([0, 1]) & groups.notna()
    return x.loc[mask], y.loc[mask], groups.loc[mask]


def _both_classes(y: pd.Series) -> bool:
    vc = y.value_counts()
    return (len(vc) >= 2) and (vc.get(0, 0) > 0) and (vc.get(1, 0) > 0)


def _group_class_coverage(y: pd.Series, groups: pd.Series):
    tmp = pd.DataFrame({"y": y.astype(int), "g": groups.astype(str)})
    gpos = tmp.loc[tmp.y == 1, "g"].nunique()
    gneg = tmp.loc[tmp.y == 0, "g"].nunique()
    return gpos, gneg


def fast_screen_ok(y: pd.Series, groups: pd.Series,
                   nmin=20, min_per_class=5, min_groups=4):
    n = len(y)
    if n < nmin:
        return False, f"n={n} < {nmin}"
    if groups.nunique() < min_groups:
        return False, f"groups={groups.nunique()} < {min_groups}"

    vc = y.value_counts()
    if not _both_classes(y):
        return False, f"single class: counts={vc.to_dict()}"
    if vc.min() < min_per_class:
        return False, f"per-class too small: counts={vc.to_dict()}, min<{min_per_class}"

    gpos, gneg = _group_class_coverage(y, groups)
    if gpos < 2 or gneg < 2:
        return False, (
            f"class-by-group segregation: pos_groups={gpos}, neg_groups={gneg} "
            f"(need >=2 each)"
        )

    return True, ""


def grouped_split(x, y, groups, test_size=test_size,
                  max_tries=32, random_state=random_state):
    def both(yy):
        vc = yy.value_counts()
        return (len(vc) >= 2) and (vc.get(0, 0) > 0) and (vc.get(1, 0) > 0)

    for i in range(max_tries):
        rs = (random_state + i) if random_state is not None else None
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=rs)
        tr, te = next(gss.split(x, y, groups))
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        if both(y_tr) and both(y_te):
            return (
                x.iloc[tr], x.iloc[te],
                y_tr, y_te,
                groups.iloc[tr], groups.iloc[te]
            )
    return None


def _adaptive_k(y, g):
    return max(2, min(5, g.nunique(), y.value_counts().min()))


def class_weights_tensor(y: pd.Series):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return torch.tensor([neg / max(1, pos)], dtype=torch.float32)


def safe_predict_proba(est, x):
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(x)
        if getattr(p, "ndim", 1) == 2 and p.shape[1] == 2:
            return p[:, 1]
        if getattr(p, "ndim", 1) == 2 and p.shape[1] == 1:
            cls = getattr(est, "classes_", None)
            n = len(x)
            if cls is not None and len(cls) == 1:
                return np.ones(n) if cls[0] == 1 else np.zeros(n)
            return p.ravel().astype(float)
        return np.asarray(p, dtype=float).ravel()

    if hasattr(est, "decision_function"):
        s = est.decision_function(x)
        s = np.asarray(s, dtype=float).ravel()
        return 1.0 / (1.0 + np.exp(-s))

    return np.asarray(est.predict(x), dtype=float).ravel()


def extract_residue_position(protein_change_val):
    if pd.isna(protein_change_val):
        return np.nan
    m = re.search(r"[A-Za-z](\d+)", str(protein_change_val))
    return int(m.group(1)) if m else np.nan


def find_labeled_file(uid_path: Path, uid: str):
    pq = uid_path / f"{uid}_Distances_Labeled.parquet"
    if pq.exists():
        return pq, "parquet"

    csv = uid_path / f"{uid}_Distances_Labeled.csv"
    if csv.exists():
        return csv, "csv"

    hits = sorted(uid_path.glob(f"{uid}_Distances_Labeled*.parquet"))
    if hits:
        return hits[0], "parquet"

    hits = sorted(uid_path.glob(f"{uid}_Distances_Labeled*.csv"))
    if hits:
        return hits[0], "csv"

    return None, None


class MLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
        super().__init__()
        layers = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(dims) - 1):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.LayerNorm(dims[i + 1]),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        x = x.float()
        return self.head(self.backbone(x)).squeeze(1)


def make_mlp(input_dim):
    net = NeuralNetClassifier(
        MLPModule,
        module__input_dim=input_dim,
        max_epochs=200,
        batch_size=64,
        optimizer=torch.optim.AdamW,
        optimizer__lr=1e-3,
        train_split=None,
        callbacks=[EarlyStopping(patience=15, monitor="train_loss", load_best=True)],
        criterion=nn.BCEWithLogitsLoss,
        device="cpu",
        verbose=0,
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", net),
    ])


def train_and_eval_mlp(uid, drug, sub_df, x, y, groups, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{uid}_{sanitize_drug_id(drug)}_mlp_{uuid.uuid4().hex[:8]}"
    print(f"[{uid} | {drug}] MLP training...")

    split = grouped_split(x, y, groups, test_size=test_size, random_state=random_state)
    if split is None:
        with open(out_dir / "split_skipped.txt", "w") as fh:
            fh.write("Could not form a valid group-aware train/test split with both classes.\n")
        print(f"  [SKIP] [{uid} | {drug}] no valid group-aware split.")
        return False

    x_tr, x_te, y_tr, y_te, g_tr, g_te = split

    if y_tr.nunique() < 2:
        with open(out_dir / "train_single_class.txt", "w") as fh:
            fh.write(f"Train split single-class: counts={y_tr.value_counts().to_dict()}\n")
        print(f"  [SKIP] [{uid} | {drug}] train split single-class.")
        return False

    pos_w = class_weights_tensor(y_tr)

    k = _adaptive_k(y_tr, g_tr)
    gkf = GroupKFold(n_splits=k)

    cv_rows = []
    fold_counter = 0

    for tr_idx, va_idx in gkf.split(x_tr, y_tr, g_tr):
        xtr, ytr = x_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        xva, yva = x_tr.iloc[va_idx], y_tr.iloc[va_idx]

        if ytr.nunique() < 2 or yva.nunique() < 2:
            print(f"  [WARN] [{uid} | {drug}] skipping CV fold with single class.")
            continue

        mlp = make_mlp(x.shape[1])
        mlp.named_steps["mlp"].set_params(
            criterion=nn.BCEWithLogitsLoss,
            criterion__pos_weight=pos_w,
        )

        mlp.fit(xtr, ytr)
        y_prob = safe_predict_proba(mlp, xva)

        thresholds = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(yva, (y_prob >= t).astype(int)) for t in thresholds]
        best_t = float(thresholds[int(np.argmax(f1s))])
        y_pred = (y_prob >= best_t).astype(int)

        meta = sub_df.loc[xva.index, ["CELL_LINE", "protein_change"]].copy()
        res_pos = meta["protein_change"].map(extract_residue_position).astype("float")

        df_fold = pd.DataFrame({
            "protein_id": uid,
            "residue_position": res_pos.values,
            "drug_id": sanitize_drug_id(drug),
            "model": "mlp",
            "observed_label": yva.values,
            "y_pred_binary": y_pred,
            "y_score": y_prob,
            "dataset_split": "oof",
            "fold_id": [f"cv{fold_counter}"] * len(yva),
            "run_id": run_id,
            "CELL_LINE": meta["CELL_LINE"].values,
            "protein_change": meta["protein_change"].values,
        }, index=xva.index)

        cv_rows.append(df_fold)
        fold_counter += 1

    mlp_final = make_mlp(x.shape[1])
    mlp_final.named_steps["mlp"].set_params(
        criterion=nn.BCEWithLogitsLoss,
        criterion__pos_weight=pos_w,
    )

    mlp_final.fit(x_tr, y_tr)
    y_prob_te = safe_predict_proba(mlp_final, x_te)
    y_pred_te = (y_prob_te >= 0.5).astype(int)

    meta_te = sub_df.loc[x_te.index, ["CELL_LINE", "protein_change"]].copy()
    res_pos_te = meta_te["protein_change"].map(extract_residue_position).astype("float")

    df_test = pd.DataFrame({
        "protein_id": uid,
        "residue_position": res_pos_te.values,
        "drug_id": sanitize_drug_id(drug),
        "model": "mlp",
        "observed_label": y_te.values,
        "y_pred_binary": y_pred_te,
        "y_score": y_prob_te,
        "dataset_split": "holdout",
        "fold_id": "",
        "run_id": run_id,
        "CELL_LINE": meta_te["CELL_LINE"].values,
        "protein_change": meta_te["protein_change"].values,
    }, index=x_te.index)

    df_pred_all = pd.concat(cv_rows + [df_test], ignore_index=True)
    df_pred_all.to_csv(out_dir / "mlp_predictions.csv", index=False)

    joblib.dump(mlp_final, out_dir / "mlp_model.joblib")

    auprc = average_precision_score(y_te, y_prob_te)
    f1 = f1_score(y_te, y_pred_te)
    cm = confusion_matrix(y_te, y_pred_te)
    brier = brier_score_loss(y_te, y_prob_te)

    with open(out_dir / "mlp_report.txt", "w") as f:
        f.write(f"Protein: {uid}\n")
        f.write(f"Drug: {drug}\n")
        f.write("Model: mlp\n")
        f.write(f"Test AUPRC: {auprc:.4f}\n")
        f.write(f"Test F1: {f1:.4f}\n")
        f.write(f"Brier: {brier:.4f}\n")
        f.write(f"n_train: {len(y_tr)}\n")
        f.write(f"n_test: {len(y_te)}\n")
        f.write(f"groups_train: {g_tr.nunique()}\n")
        f.write(f"groups_test: {g_te.nunique()}\n")
        f.write("\nConfusion matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification report:\n")
        f.write(classification_report(y_te, y_pred_te, digits=4, zero_division=0))

    print(f"[{uid} | {drug}] MLP done | AUPRC={auprc:.3f} | F1={f1:.3f}")
    gc.collect()
    return True


def process_protein(uid_path: Path):
    uid = uid_path.name

    file_path, file_type = find_labeled_file(uid_path, uid)
    if file_path is None:
        print(f"[SKIP] {uid}: no labeled parquet/csv found.")
        return 0, 1

    try:
        if file_type == "parquet":
            df = pd.read_parquet(file_path)
            print(f"[INFO] {uid}: loaded parquet ({file_path.name})")
        else:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"[INFO] {uid}: loaded csv ({file_path.name})")
    except Exception as e:
        print(f"[ERROR] {uid}: load failed -> {e}")
        return 0, 1

    df = ensure_clust_dummies(df)

    if "drug" not in df.columns or "sensitivity" not in df.columns or "CELL_LINE" not in df.columns:
        print(f"[SKIP] {uid}: missing required columns.")
        return 0, 1

    feat_cols = pick_feature_columns(df)
    if not feat_cols:
        print(f"[SKIP] {uid}: no valid numeric feature columns.")
        return 0, 1

    drugs = sorted(df["drug"].dropna().astype(str).unique())
    if target_drug is not None:
        drugs = [d for d in drugs if str(d) == str(target_drug)]

    if not drugs:
        print(f"[SKIP] {uid}: no drugs to process.")
        return 0, 1

    if use_clust and not any(c.startswith("clust_") for c in df.columns):
        print(f"[WARN] {uid}: use_clust=True but no clust_* features were found.")

    print(f"\n=== {uid} | drugs={len(drugs)} | features={len(feat_cols)} ===")

    trained = 0
    skipped = 0

    for drug in drugs:
        out_dir = mlp_output_root / uid / sanitize_drug_id(drug)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Writing outputs to: {out_dir.resolve()}")

        sub = df[df["drug"].astype(str) == str(drug)].copy()

        try:
            x, y, groups = prepare_xy_groups(sub, feat_cols)
        except SystemExit as e:
            print(f"  [SKIP] [{uid} | {drug}] {e}")
            skipped += 1
            continue

        ok, reason = fast_screen_ok(
            y,
            groups,
            nmin=min_samples_total,
            min_per_class=min_samples_per_class,
            min_groups=min_groups,
        )

        if not ok:
            with open(out_dir / "screen_skipped.txt", "w") as fh:
                fh.write(f"Fast screen failed: {reason}\n")
            print(f"  [SKIP] [{uid} | {drug}] {reason}")
            skipped += 1
            continue

        success = train_and_eval_mlp(uid, drug, sub, x, y, groups, out_dir)
        if success:
            trained += 1
        else:
            skipped += 1

    return trained, skipped

# Run main
def main():
    if target_protein is None:
        print("Usage: python script.py UNIPROT_ID")
        sys.exit(1)

    uid_path = dist_root / target_protein

    if not uid_path.exists() or not uid_path.is_dir():
        print(f"[WARN] Protein folder not found: {uid_path}")
        sys.exit(1)

    print(f"[INFO] Processing protein: {target_protein}")

    trained, skipped = process_protein(uid_path)

    print(f"\n[DONE] MLP training complete for {target_protein}. Trained pairs: {trained} | Skipped: {skipped}")


if __name__ == "__main__":
    main()