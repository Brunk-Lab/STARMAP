from __future__ import annotations

import gc
import re
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, brier_score_loss, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    from skorch import NeuralNetClassifier
    from skorch.callbacks import EarlyStopping
except Exception:  # pragma: no cover
    torch = None
    nn = None
    NeuralNetClassifier = None
    EarlyStopping = None

from starmap.io import sanitize_name

PATTERNS = [
    re.compile(r"^DTAS_(\d+)$"), re.compile(r"^DTBS_(\d+)$"),
    re.compile(r"^AS_(\d+)_(x|y|z)$"), re.compile(r"^BS_(\d+)_(x|y|z)$"),
    re.compile(r"^DTASR_(\d+)-(\d+)$"), re.compile(r"^DTBSR_(\d+)-(\d+)$"),
    re.compile(r"^ASR_(\d+)-(\d+)_(x|y|z)$"), re.compile(r"^BSR_(\d+)-(\d+)_(x|y|z)$"),
    re.compile(r"^Mut_CA_(x|y|z)$"),
]
EXCLUDE_COLS = {"ID", "CELL_LINE", "SIFT", "LIKELY_LOF", "protein_change", "drug", "AUC_CTRP", "sensitivity", "clust"}


def ensure_clust_dummies(df: pd.DataFrame, use_clust: bool = True) -> pd.DataFrame:
    if not use_clust or "clust" not in df.columns:
        return df
    cl = df["clust"]
    if pd.api.types.is_numeric_dtype(cl):
        cl_str = cl.astype("Int64").astype(str).radd("c")
    else:
        cl_str = cl.astype(str).apply(lambda s: f"c{s}" if s and str(s).lower() != "nan" else "c_missing")
    cl_str = cl_str.fillna("c_missing").replace({"nan": "c_missing", "c<NA>": "c_missing"})
    return pd.concat([df, pd.get_dummies(cl_str, prefix="clust", dtype=float)], axis=1)


def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if c.startswith("clust_") or any(p.match(c) for p in PATTERNS):
            cols.append(c)
    return sorted(cols)


def normalize_labels(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    mapped = s.map({"sensitive": 1.0, "not sensitive": 0.0})
    return mapped.dropna().astype(float)


def prepare_xy_groups(df: pd.DataFrame, feat_cols: list[str]):
    y = normalize_labels(df["sensitivity"])
    x = df.loc[:, feat_cols].copy()
    groups = df.get("CELL_LINE", pd.Series(["_NA_"] * len(df), index=df.index)).astype(str)
    mask = x.notna().all(axis=1) & y.isin([0, 1]) & groups.notna()
    return x.loc[mask], y.loc[mask], groups.loc[mask]


def fast_screen_ok(y: pd.Series, groups: pd.Series, nmin=20, min_per_class=5, min_groups=4):
    if len(y) < nmin:
        return False, f"n={len(y)} < {nmin}"
    if groups.nunique() < min_groups:
        return False, f"groups={groups.nunique()} < {min_groups}"
    vc = y.value_counts()
    if len(vc) < 2 or vc.min() < min_per_class:
        return False, f"class counts too small: {vc.to_dict()}"
    tmp = pd.DataFrame({"y": y.astype(int), "g": groups.astype(str)})
    if tmp.loc[tmp.y == 1, "g"].nunique() < 2 or tmp.loc[tmp.y == 0, "g"].nunique() < 2:
        return False, "class-by-group segregation"
    return True, ""


def grouped_split(x, y, groups, test_size=0.30, max_tries=32, random_state=42):
    for i in range(max_tries):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + i)
        tr, te = next(gss.split(x, y, groups))
        if y.iloc[tr].nunique() == 2 and y.iloc[te].nunique() == 2:
            return x.iloc[tr], x.iloc[te], y.iloc[tr], y.iloc[te], groups.iloc[tr], groups.iloc[te]
    return None


def extract_residue_position(value):
    if pd.isna(value):
        return np.nan
    m = re.search(r"[A-Za-z](\d+)", str(value))
    return int(m.group(1)) if m else np.nan


if nn is not None:
    class MLPModule(nn.Module):
        def __init__(self, input_dim, hidden_dims=(64, 32), dropout=0.3):
            super().__init__()
            layers = []
            dims = [input_dim, *hidden_dims]
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.LayerNorm(dims[i + 1]), nn.Dropout(dropout)]
            self.backbone = nn.Sequential(*layers)
            self.head = nn.Linear(hidden_dims[-1], 1)
        def forward(self, x):
            return self.head(self.backbone(x.float())).squeeze(1)
else:
    MLPModule = None


def make_mlp(input_dim: int, max_epochs: int = 200, batch_size: int = 64, lr: float = 1e-3):
    if NeuralNetClassifier is None:
        raise RuntimeError("torch and skorch are required for MLP training")
    net = NeuralNetClassifier(
        MLPModule,
        module__input_dim=input_dim,
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=torch.optim.AdamW,
        optimizer__lr=lr,
        train_split=None,
        callbacks=[EarlyStopping(patience=15, monitor="train_loss", load_best=True)],
        criterion=nn.BCEWithLogitsLoss,
        device="cpu",
        verbose=0,
    )
    return Pipeline([("scaler", StandardScaler()), ("mlp", net)])


def safe_predict_proba(est, x):
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(x)
        if getattr(p, "ndim", 1) == 2 and p.shape[1] >= 2:
            return p[:, 1]
        return np.asarray(p).ravel().astype(float)
    if hasattr(est, "decision_function"):
        return expit(np.asarray(est.decision_function(x)).ravel().astype(float))
    return np.asarray(est.predict(x)).ravel().astype(float)


def train_mlp_for_drug(uid: str, drug: str, sub_df: pd.DataFrame, x, y, groups, out_dir: str | Path, test_size=0.30, random_state=42, max_epochs=200) -> bool:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    split = grouped_split(x, y, groups, test_size=test_size, random_state=random_state)
    if split is None:
        (out_dir / "split_skipped.txt").write_text("Could not form a valid group-aware split.\n")
        return False
    x_tr, x_te, y_tr, y_te, g_tr, g_te = split
    k = max(2, min(5, g_tr.nunique(), y_tr.value_counts().min()))
    pos = (y_tr == 1).sum(); neg = (y_tr == 0).sum()
    pos_w = torch.tensor([neg / max(1, pos)], dtype=torch.float32)
    run_id = f"{uid}_{sanitize_name(drug)}_mlp_{uuid.uuid4().hex[:8]}"
    cv_rows = []
    for fold, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=k).split(x_tr, y_tr, g_tr)):
        xtr, xva = x_tr.iloc[tr_idx], x_tr.iloc[va_idx]
        ytr, yva = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
        if ytr.nunique() < 2 or yva.nunique() < 2:
            continue
        model = make_mlp(x.shape[1], max_epochs=max_epochs)
        model.named_steps["mlp"].set_params(criterion=nn.BCEWithLogitsLoss, criterion__pos_weight=pos_w)
        model.fit(xtr, ytr)
        prob = safe_predict_proba(model, xva)
        thresholds = np.linspace(0.05, 0.95, 19)
        best_t = float(thresholds[int(np.argmax([f1_score(yva, prob >= t) for t in thresholds]))])
        pred = (prob >= best_t).astype(int)
        meta = sub_df.loc[xva.index, ["CELL_LINE", "protein_change"]].copy()
        cv_rows.append(pd.DataFrame({
            "protein_id": uid, "residue_position": meta["protein_change"].map(extract_residue_position).values,
            "drug_id": sanitize_name(drug), "model": "mlp", "observed_label": yva.values,
            "y_pred_binary": pred, "y_score": prob, "dataset_split": "oof", "fold_id": f"cv{fold}",
            "run_id": run_id, "CELL_LINE": meta["CELL_LINE"].values, "protein_change": meta["protein_change"].values,
        }, index=xva.index))
    final = make_mlp(x.shape[1], max_epochs=max_epochs)
    final.named_steps["mlp"].set_params(criterion=nn.BCEWithLogitsLoss, criterion__pos_weight=pos_w)
    final.fit(x_tr, y_tr)
    prob = safe_predict_proba(final, x_te); pred = (prob >= 0.5).astype(int)
    meta = sub_df.loc[x_te.index, ["CELL_LINE", "protein_change"]].copy()
    test_df = pd.DataFrame({
        "protein_id": uid, "residue_position": meta["protein_change"].map(extract_residue_position).values,
        "drug_id": sanitize_name(drug), "model": "mlp", "observed_label": y_te.values,
        "y_pred_binary": pred, "y_score": prob, "dataset_split": "holdout", "fold_id": "",
        "run_id": run_id, "CELL_LINE": meta["CELL_LINE"].values, "protein_change": meta["protein_change"].values,
    }, index=x_te.index)
    pd.concat(cv_rows + [test_df], ignore_index=True).to_csv(out_dir / "mlp_predictions.csv", index=False)
    joblib.dump(final, out_dir / "mlp_model.joblib")
    with open(out_dir / "mlp_report.txt", "w") as f:
        f.write(f"Protein: {uid}\nDrug: {drug}\n")
        f.write(f"Test AUPRC: {average_precision_score(y_te, prob):.4f}\n")
        f.write(f"Test F1: {f1_score(y_te, pred):.4f}\n")
        f.write(f"Brier: {brier_score_loss(y_te, prob):.4f}\n")
        f.write("\nConfusion matrix:\n" + str(confusion_matrix(y_te, pred)))
        f.write("\n\nClassification report:\n" + classification_report(y_te, pred, digits=4, zero_division=0))
    gc.collect()
    return True


def run_mlp_for_uniprot(uniprot_id: str, dist_root: str | Path, output_root: str | Path, drug: str | None = None, use_clust: bool = True, max_epochs: int = 200) -> dict[str, int]:
    uid = uniprot_id.upper().strip()
    folder = Path(dist_root) / uid
    pq = folder / f"{uid}_Distances_Labeled.parquet"
    csv = folder / f"{uid}_Distances_Labeled.csv"
    if pq.exists():
        df = pd.read_parquet(pq)
    elif csv.exists():
        df = pd.read_csv(csv, low_memory=False)
    else:
        raise FileNotFoundError(f"No labeled distance table found under {folder}")
    df = ensure_clust_dummies(df, use_clust=use_clust)
    feat_cols = pick_feature_columns(df)
    if not feat_cols:
        raise ValueError("No valid numeric feature columns found")
    drugs = sorted(df["drug"].dropna().astype(str).unique())
    if drug is not None:
        drugs = [d for d in drugs if d == str(drug)]
    trained = skipped = 0
    for d in drugs:
        out_dir = Path(output_root) / uid / sanitize_name(d)
        sub = df[df["drug"].astype(str) == d].copy()
        x, y, groups = prepare_xy_groups(sub, feat_cols)
        ok, reason = fast_screen_ok(y, groups)
        if not ok:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "screen_skipped.txt").write_text(reason + "\n")
            skipped += 1; continue
        if train_mlp_for_drug(uid, d, sub, x, y, groups, out_dir, max_epochs=max_epochs):
            trained += 1
        else:
            skipped += 1
    return {"trained": trained, "skipped": skipped}
