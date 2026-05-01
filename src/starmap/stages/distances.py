from __future__ import annotations

import ast
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


def safe_eval_list(s, default=None):
    default = [] if default is None else default
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return default
    if isinstance(s, list):
        return s
    text = str(s).strip()
    if not text:
        return default
    for loader in (json.loads, ast.literal_eval):
        try:
            val = loader(text)
            return val if isinstance(val, list) else default
        except Exception:
            pass
    return default


def _as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _coerce_int_list(x):
    if not isinstance(x, (list, tuple, set)):
        return []
    out = []
    for v in x:
        p = _as_int(v)
        if p is not None:
            out.append(p)
    return sorted(set(out))


def safe_eval_bindings(s):
    singles, ranges = [], []
    for item in safe_eval_list(s, []):
        if not isinstance(item, dict):
            continue
        if ("positions" in item) or ("start" in item) or ("end" in item) or item.get("is_range"):
            pos_list = _coerce_int_list(item.get("positions"))
            if not pos_list:
                start = _as_int(item.get("start")); end = _as_int(item.get("end"))
                if start is not None and end is not None:
                    if end < start:
                        start, end = end, start
                    pos_list = list(range(start, end + 1))
                else:
                    p = _as_int(item.get("position"))
                    if p is not None:
                        pos_list = [p]
            if pos_list:
                ranges.append((min(pos_list), max(pos_list), pos_list))
        else:
            p = _as_int(item.get("position"))
            if p is not None:
                singles.append((p,))
    return singles, ranges


def normalize_uniprot_id(raw_id):
    if raw_id is None or (isinstance(raw_id, float) and np.isnan(raw_id)):
        return None
    try:
        v = ast.literal_eval(str(raw_id))
        if isinstance(v, list) and v:
            raw_id = str(v[0])
    except Exception:
        pass
    s = str(raw_id).strip()
    if s.startswith("AF-") and "-F" in s:
        s = s.replace("AF-", "").split("-F")[0]
    return s or None


def download_alphafold_pdb(uniprot_id: str, outdir: str | Path, overwrite: bool = False) -> Path | None:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    pdb_path = outdir / f"{uniprot_id}.pdb"
    if pdb_path.exists() and not overwrite:
        return pdb_path
    if requests is None:
        raise RuntimeError("requests is required to download AlphaFold PDB files")
    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    r = requests.get(api_url, timeout=60); r.raise_for_status()
    data = r.json()
    if not data or not data[0].get("pdbUrl"):
        return None
    rr = requests.get(data[0]["pdbUrl"], timeout=60); rr.raise_for_status()
    pdb_path.write_bytes(rr.content)
    return pdb_path


def build_ca_map(pdb_path: str | Path) -> dict[int, np.ndarray]:
    structure = PDBParser(QUIET=True).get_structure("AF", str(pdb_path))
    ca_map = {}
    for res in structure.get_residues():
        try:
            resseq = int(res.id[1])
        except Exception:
            continue
        if "CA" in res:
            ca_map[resseq] = np.array(res["CA"].coord, dtype=float)
    return ca_map


def _centroid(pos_list, ca_map):
    pts = [ca_map[p] for p in pos_list if p in ca_map]
    return np.vstack(pts).mean(axis=0) if pts else None


def _distance(a, b):
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def _compute_one(row, ca_map, actives, binding_singles, binding_ranges):
    entry = {k: row.get(k) for k in ["ID", "CELL_LINE", "SIFT", "LIKELY_LOF", "protein_change"]}
    pos = _as_int(row.get("AA_POS"))
    if pos is None or pos not in ca_map:
        return None
    mut_xyz = ca_map[pos]
    entry.update({"Mut_CA_x": mut_xyz[0], "Mut_CA_y": mut_xyz[1], "Mut_CA_z": mut_xyz[2]})
    for p in actives:
        if p in ca_map:
            xyz = ca_map[p]
            entry[f"DTAS_{p}"] = _distance(mut_xyz, xyz)
            entry[f"AS_{p}_x"], entry[f"AS_{p}_y"], entry[f"AS_{p}_z"] = xyz
    for p in binding_singles:
        if p in ca_map:
            xyz = ca_map[p]
            entry[f"DTBS_{p}"] = _distance(mut_xyz, xyz)
            entry[f"BS_{p}_x"], entry[f"BS_{p}_y"], entry[f"BS_{p}_z"] = xyz
    for start, end, cen in binding_ranges:
        if cen is not None:
            key = f"{start}-{end}"
            entry[f"DTBSR_{key}"] = _distance(mut_xyz, cen)
            entry[f"BSR_{key}_x"], entry[f"BSR_{key}_y"], entry[f"BSR_{key}_z"] = cen
    return entry


def make_distance_table(mutations: pd.DataFrame, pdb_path: str | Path) -> pd.DataFrame:
    ca_map = build_ca_map(pdb_path)
    if not ca_map:
        raise ValueError(f"No C-alpha coordinates parsed from {pdb_path}")
    active_union, binding_pos_union, binding_ranges_raw = set(), set(), []
    for _, r in mutations.iterrows():
        for pos in safe_eval_list(r.get("ActiveSitePositions"), []):
            p = _as_int(pos)
            if p is not None:
                active_union.add(p)
        singles, ranges = safe_eval_bindings(r.get("BindingSites"))
        binding_pos_union.update(p for (p,) in singles)
        binding_ranges_raw.extend(ranges)
    seen, ranges = set(), []
    for start, end, pos_list in binding_ranges_raw:
        key = (start, end, tuple(pos_list))
        if key not in seen:
            seen.add(key); ranges.append((start, end, _centroid(pos_list, ca_map)))
    rows = []
    for _, row in mutations.iterrows():
        out = _compute_one(row.to_dict(), ca_map, sorted(active_union), sorted(binding_pos_union), ranges)
        if out is not None:
            rows.append(out)
    return pd.DataFrame(rows)


def run_distance_generation(uniprot_id: str, input_csv: str | Path, output_dir: str | Path, alphafold_id_col: str = "AlphaFold_IDs", pdb_path: str | Path | None = None) -> Path:
    uniprot_id = normalize_uniprot_id(uniprot_id)
    df = pd.read_csv(input_csv)
    df["uniprot_id"] = df[alphafold_id_col].apply(normalize_uniprot_id)
    sub = df[df["uniprot_id"] == uniprot_id].copy()
    if sub.empty:
        raise ValueError(f"No rows found for UniProt ID {uniprot_id}")
    outdir = Path(output_dir) / uniprot_id
    outdir.mkdir(parents=True, exist_ok=True)
    pdb = Path(pdb_path) if pdb_path else download_alphafold_pdb(uniprot_id, outdir)
    if pdb is None:
        raise FileNotFoundError(f"No AlphaFold PDB found for {uniprot_id}")
    out_df = make_distance_table(sub, pdb)
    out_csv = outdir / f"{uniprot_id}_Distances.csv"
    out_df.to_csv(out_csv, index=False)
    return out_csv
