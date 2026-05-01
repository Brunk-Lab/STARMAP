# Import necessary packages
import os, ast, json, math, re, sys
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
try:
    import requests
    HAVE_REQUESTS = True
except Exception:
    HAVE_REQUESTS = False

# Config
input_csv = "../data/ParsedMutations_Plus_Sites_Filtered.csv"
out_dir = "../output/dist_files"

# Read UniProt ID from command line:
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py UNIPROT_ID")
    sys.exit(1)
    
target_protein = sys.argv[1].upper().strip()


# -------------------------
# Helper functions
# -------------------------
def safe_eval_list(s, default):
    if s is None:
        return default
    if isinstance(s, list):
        return s
    text = str(s).strip()
    if not text:
        return default
    try:
        val = json.loads(text)
        return val if isinstance(val, list) else default
    except Exception:
        pass
    try:
        val = ast.literal_eval(text)
        return val if isinstance(val, list) else default
    except Exception:
        return default


def _as_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _coerce_int_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        out = []
        for v in x:
            try:
                out.append(int(v))
            except Exception:
                pass
        return sorted(set(out))
    return []


def safe_eval_bindings(s):
    singles, ranges = [], []
    for item in safe_eval_list(s, []):
        if not isinstance(item, dict):
            continue
        if ("positions" in item) or ("start" in item) or ("end" in item) or item.get("is_range"):
            pos_list = _coerce_int_list(item.get("positions"))
            if not pos_list:
                start = _as_int(item.get("start"))
                end = _as_int(item.get("end"))
                if start is not None and end is not None:
                    if end < start:
                        start, end = end, start
                    pos_list = list(range(start, end + 1))
                else:
                    p = _as_int(item.get("position"))
                    if p is not None:
                        pos_list = [p]
            if pos_list:
                start, end = min(pos_list), max(pos_list)
                ranges.append((start, end, pos_list))
        else:
            p = _as_int(item.get("position"))
            if p is not None:
                singles.append((p,))
    return singles, ranges


def distance(a, b):
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def normalize_uniprot_id(raw_id):
    """
    Accepts things like:
      P01116
      AF-P01116-F1
      ['P01116']
      ['AF-P01116-F1']
    Returns:
      P01116
    """
    if raw_id is None:
        return None

    try:
        v = ast.literal_eval(raw_id)
        if isinstance(v, list) and len(v) > 0:
            raw_id = str(v[0])
    except Exception:
        pass

    s = str(raw_id).strip()
    if not s:
        return None

    if s.startswith("AF-") and "-F" in s:
        s = s.replace("AF-", "").split("-F")[0]

    return s


def download_alphafold_pdb(uniprot_id: str, outdir: Path):
    """
    Download AlphaFold PDB for a UniProt accession using the AlphaFold API.
    Saves as {uniprot_id}.pdb
    """
    pdb_path = outdir / f"{uniprot_id}.pdb"

    if pdb_path.exists():
        print(f"    Found existing PDB: {pdb_path}")
        return pdb_path

    if not HAVE_REQUESTS:
        print(f"    requests not available; cannot download {uniprot_id}. Skipping.")
        return None

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"

    try:
        print(f"    Querying AlphaFold API for {uniprot_id}...")
        r = requests.get(api_url, timeout=60)
        r.raise_for_status()
        data = r.json()

        if not data:
            print(f"    No AlphaFold entry found for {uniprot_id}.")
            return None

        rec = data[0]

        pdb_url = rec.get("pdbUrl")
        if not pdb_url:
            print(f"    No PDB URL found for {uniprot_id} in AlphaFold API response.")
            return None

        print(f"    Downloading PDB from {pdb_url} ...")
        rr = requests.get(pdb_url, timeout=60)
        rr.raise_for_status()

        pdb_path.write_bytes(rr.content)
        print(f"    Saved PDB to {pdb_path}")
        return pdb_path

    except Exception as e:
        print(f"    Failed to download AlphaFold PDB for {uniprot_id}: {e}")
        return None


def build_ca_map(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("AF", str(pdb_path))
    ca_map = {}
    for res in structure.get_residues():
        try:
            resseq = int(res.id[1])
        except Exception:
            continue
        if "CA" in res:
            ca_map[resseq] = np.array(res["CA"].coord, dtype=float)
    return ca_map


def _centroid_from_positions(pos_list, ca_map):
    pts = []
    for p in pos_list:
        xyz = ca_map.get(p)
        if xyz is not None:
            pts.append(xyz)
    if not pts:
        return None
    arr = np.vstack(pts).astype(float)
    return arr.mean(axis=0)


_G_CA_MAP = None
_G_ACTIVES = None
_G_BINDINGS_POS = None
_G_BINDING_RANGE_CENTROIDS = None


def _init_worker(ca_map, actives_sorted, bindings_pos_sorted, binding_range_centroids):
    global _G_CA_MAP, _G_ACTIVES, _G_BINDINGS_POS, _G_BINDING_RANGE_CENTROIDS
    _G_CA_MAP = ca_map
    _G_ACTIVES = actives_sorted
    _G_BINDINGS_POS = bindings_pos_sorted
    _G_BINDING_RANGE_CENTROIDS = binding_range_centroids


def _compute_row(row_dict):
    global _G_CA_MAP, _G_ACTIVES, _G_BINDINGS_POS, _G_BINDING_RANGE_CENTROIDS

    entry = {
        "ID": row_dict.get("ID"),
        "CELL_LINE": row_dict.get("CELL_LINE"),
        "SIFT": row_dict.get("SIFT"),
        "LIKELY_LOF": row_dict.get("LIKELY_LOF"),
        "protein_change": row_dict.get("protein_change"),
    }

    try:
        mut_pos = int(row_dict.get("AA_POS"))
    except Exception:
        return None

    mut_xyz = _G_CA_MAP.get(mut_pos)
    if mut_xyz is None:
        return None

    entry["Mut_CA_x"] = float(mut_xyz[0])
    entry["Mut_CA_y"] = float(mut_xyz[1])
    entry["Mut_CA_z"] = float(mut_xyz[2])

    for pos in _G_ACTIVES or []:
        site_xyz = _G_CA_MAP.get(pos)
        if site_xyz is None:
            continue
        entry[f"DTAS_{pos}"] = distance(mut_xyz, site_xyz)
        entry[f"AS_{pos}_x"] = float(site_xyz[0])
        entry[f"AS_{pos}_y"] = float(site_xyz[1])
        entry[f"AS_{pos}_z"] = float(site_xyz[2])

    for (pos,) in _G_BINDINGS_POS or []:
        site_xyz = _G_CA_MAP.get(pos)
        if site_xyz is None:
            continue
        entry[f"DTBS_{pos}"] = distance(mut_xyz, site_xyz)
        entry[f"BS_{pos}_x"] = float(site_xyz[0])
        entry[f"BS_{pos}_y"] = float(site_xyz[1])
        entry[f"BS_{pos}_z"] = float(site_xyz[2])

    for (start, end, centroid_xyz, _used_positions) in _G_BINDING_RANGE_CENTROIDS or []:
        if centroid_xyz is None:
            continue
        key = f"{start}-{end}"
        entry[f"DTBSR_{key}"] = distance(mut_xyz, centroid_xyz)
        entry[f"BSR_{key}_x"] = float(centroid_xyz[0])
        entry[f"BSR_{key}_y"] = float(centroid_xyz[1])
        entry[f"BSR_{key}_z"] = float(centroid_xyz[2])

    return entry


# -------------------------
# Main
# -------------------------
def main():
    df = pd.read_csv(input_csv)

    # Add normalized UniProt column from AlphaFold_IDs
    df["uniprot_id"] = df["AlphaFold_IDs"].apply(normalize_uniprot_id)

    if target_protein is not None:
        target = normalize_uniprot_id(target_protein)
        df = df[df["uniprot_id"] == target].copy()

        if df.empty:
            print(f"No rows found for UniProt ID: {target}")
            return

        print(f"Filtering to UniProt ID: {target}")

    unique_uniprots = df["uniprot_id"].dropna().unique()
    total_ids = len(unique_uniprots)
    print(f"Found {total_ids} UniProt IDs in input.")

    for idx, uniprot_id in enumerate(unique_uniprots, start=1):
        sub = df[df["uniprot_id"] == uniprot_id].copy()
        n_rows = len(sub)
        print(f"\n[{idx}/{total_ids}] Processing {uniprot_id} ({n_rows} mutations)")

        outdir = Path(out_dir) / uniprot_id
        outdir.mkdir(parents=True, exist_ok=True)

        pdb_path = download_alphafold_pdb(uniprot_id, outdir)
        if pdb_path is None:
            print(f"    Missing PDB for {uniprot_id}. Skipping.")
            continue

        ca_map = build_ca_map(pdb_path)
        if not ca_map:
            print(f"    No C-alpha coords parsed for {uniprot_id}. Skipping.")
            continue
        print(f"    Parsed {len(ca_map)} residues with C-alpha coords.")

        active_union = set()
        binding_pos_union = set()
        binding_ranges_union = []

        for _, r in sub.iterrows():
            for pos in safe_eval_list(r.get("ActiveSitePositions"), []):
                p = _as_int(pos)
                if p is not None:
                    active_union.add(p)

            singles, ranges = safe_eval_bindings(r.get("BindingSites"))
            for (p,) in singles:
                binding_pos_union.add(p)
            binding_ranges_union.extend(ranges)

        all_actives_sorted = sorted(active_union)
        all_bindings_pos_sorted = sorted((p,) for p in binding_pos_union)

        seen_rg, all_binding_ranges_sorted = set(), []
        for start, end, pos_list in binding_ranges_union:
            key = (start, end, tuple(pos_list))
            if key not in seen_rg:
                seen_rg.add(key)
                all_binding_ranges_sorted.append((start, end, pos_list))
        all_binding_ranges_sorted.sort(key=lambda x: (x[0], x[1]))

        binding_range_centroids = []
        have_centroids = 0
        for start, end, pos_list in all_binding_ranges_sorted:
            cen = _centroid_from_positions(pos_list, ca_map)
            if cen is not None:
                have_centroids += 1
            binding_range_centroids.append((start, end, cen, pos_list))

        print(f"    Active sites (unique): {len(all_actives_sorted)}")
        print(f"    Binding singles (unique positions): {len(all_bindings_pos_sorted)}")
        print(
            f"    Binding ranges (unique spans): {len(all_binding_ranges_sorted)} "
            f"(centroids computed: {have_centroids})"
        )

        rows = [sub.iloc[i].to_dict() for i in range(n_rows)]

        from concurrent.futures import ProcessPoolExecutor, as_completed
        results = []

        with ProcessPoolExecutor(
            initializer=_init_worker,
            initargs=(ca_map, all_actives_sorted, all_bindings_pos_sorted, binding_range_centroids),
        ) as ex:
            futures = [ex.submit(_compute_row, row) for row in rows]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    results.append(res)

        if not results:
            print(f"    No rows produced for {uniprot_id}.")
            continue

        base_cols = ["ID", "CELL_LINE", "SIFT", "LIKELY_LOF", "protein_change"]
        out_df = pd.DataFrame(results)

        acp_positions = sorted({
            int(c.split("_", 1)[1])
            for c in out_df.columns if c.startswith("DTAS_")
        })
        bs_positions = sorted({
            int(c.split("_", 1)[1])
            for c in out_df.columns if c.startswith("DTBS_")
        })

        def _parse_range_key(col):
            try:
                rng = col.split("_", 1)[1]
                a, b = rng.split("-", 1)
                return (int(a), int(b))
            except Exception:
                return None

        bsr_keys = sorted({
            _parse_range_key(c) for c in out_df.columns if c.startswith("DTBSR_")
        } - {None})

        cols = (
            base_cols
            + ["Mut_CA_x", "Mut_CA_y", "Mut_CA_z"]
            + [f"DTAS_{p}" for p in acp_positions]
            + [f"AS_{p}_x" for p in acp_positions]
            + [f"AS_{p}_y" for p in acp_positions]
            + [f"AS_{p}_z" for p in acp_positions]
            + [f"DTBS_{p}" for p in bs_positions]
            + [f"BS_{p}_x" for p in bs_positions]
            + [f"BS_{p}_y" for p in bs_positions]
            + [f"BS_{p}_z" for p in bs_positions]
            + [f"DTBSR_{s}-{e}" for (s, e) in bsr_keys]
            + [f"BSR_{s}-{e}_x" for (s, e) in bsr_keys]
            + [f"BSR_{s}-{e}_y" for (s, e) in bsr_keys]
            + [f"BSR_{s}-{e}_z" for (s, e) in bsr_keys]
        )

        for c in cols:
            if c not in out_df.columns:
                out_df[c] = np.nan
        out_df = out_df[cols]

        out_csv = Path(out_dir) / uniprot_id / f"{uniprot_id}_Distances.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"    -> Wrote {out_csv} ({len(out_df)} rows, {len(out_df.columns)} cols)")


if __name__ == "__main__":
    main()