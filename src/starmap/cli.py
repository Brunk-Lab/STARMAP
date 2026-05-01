from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _legacy_script(name: str) -> Path:
    return Path(__file__).parent / "legacy" / name


def run_legacy(script_name: str, arg: str) -> None:
    script = _legacy_script(script_name)
    old_argv = sys.argv[:]
    sys.argv = [str(script), arg]
    try:
        runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = old_argv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="starmap", description="Run STARMAP pipeline stages.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("nmf", help="Run NMF flatmap generation for one gene")
    p.add_argument("gene")
    p.add_argument("--coord-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-components", type=int, default=3)
    p.add_argument("--max-components", type=int, default=6)
    p.add_argument("--random-state", type=int, default=0)

    p = sub.add_parser("distances", help="Download/read AlphaFold PDB and compute mutation-to-site distances")
    p.add_argument("uniprot_id")
    p.add_argument("--input-csv", required=True, help="Parsed mutations CSV")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--alphafold-id-col", default="AlphaFold_IDs")
    p.add_argument("--pdb-path", default=None, help="Optional local PDB path; skips AlphaFold download")

    p = sub.add_parser("annotate", help="Add CTRP sensitivity labels and cluster IDs to a distance table")
    p.add_argument("uniprot_id")
    p.add_argument("--ccle-csv", required=True)
    p.add_argument("--dist-dir", required=True)
    p.add_argument("--map-csv", default=None)
    p.add_argument("--output-suffix", default="_Distances_Labeled.parquet")
    p.add_argument("--sensitivity-quantile", type=float, default=0.25)

    p = sub.add_parser("mlp", help="Train MLP drug-response models for one UniProt ID")
    p.add_argument("uniprot_id")
    p.add_argument("--dist-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--drug", default=None)
    p.add_argument("--no-clust", action="store_true")
    p.add_argument("--max-epochs", type=int, default=200)

    p = sub.add_parser("logodds", help="Run cluster-level drug log-odds for one UniProt ID")
    p.add_argument("uniprot_id")
    p.add_argument("--mlp-root", required=True)
    p.add_argument("--nmf-root", required=True)
    p.add_argument("--mapping-csv", required=True)
    p.add_argument("--output-root", required=True)

    p = sub.add_parser("gistar", help="Run Getis-Ord Gi* TRN scoring for one gene")
    p.add_argument("gene")
    p.add_argument("--gsea-dir", required=True)
    p.add_argument("--position-cellline-csv", required=True)
    p.add_argument("--nmf-root", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--no-gzip", action="store_true")

    p = sub.add_parser("tahoe-confidence", help="Generate Tahoe empirical confidence file for one gene")
    p.add_argument("gene")
    p.add_argument("--sorted-logodds-csv", required=True)
    p.add_argument("--perdrug-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pvalue-cutoff", type=float, default=0.05)

    legacy = {
        "download-perturbseq": "download_perturbseq_data.py",
        "preprocess-perturbseq": "preprocess_perturbseq_data.py",
        "perturbseq-gene-output": "genespecific_perturbseq_output.py",
        "perturbseq-confidence": "generate_confidence_scores.py",
        "tahoe-expression-legacy": "perdrug_expression_mutvother.py",
    }
    for cmd, script in legacy.items():
        q = sub.add_parser(cmd, help=f"Run legacy script: {script}")
        q.add_argument("argument", help="GENE or UNIPROT_ID, depending on the script")

    args = parser.parse_args(argv)

    if args.cmd == "nmf":
        from starmap.stages.nmf import run_nmf_for_gene
        out = run_nmf_for_gene(args.gene, args.coord_csv, args.output_dir, args.min_components, args.max_components, args.random_state)
    elif args.cmd == "distances":
        from starmap.stages.distances import run_distance_generation
        out = run_distance_generation(args.uniprot_id, args.input_csv, args.output_dir, args.alphafold_id_col, args.pdb_path)
    elif args.cmd == "annotate":
        from starmap.stages.annotate import run_annotate_auc_cluster
        out = run_annotate_auc_cluster(args.uniprot_id, args.ccle_csv, args.dist_dir, args.map_csv, args.output_suffix, args.sensitivity_quantile)
    elif args.cmd == "mlp":
        from starmap.stages.mlp import run_mlp_for_uniprot
        res = run_mlp_for_uniprot(args.uniprot_id, args.dist_root, args.output_root, args.drug, not args.no_clust, args.max_epochs)
        print(res)
        return 0
    elif args.cmd == "logodds":
        from starmap.stages.logodds import run_logodds_for_uniprot
        out = run_logodds_for_uniprot(args.uniprot_id, args.mlp_root, args.nmf_root, args.mapping_csv, args.output_root)
    elif args.cmd == "gistar":
        from starmap.stages.gistar import run_gistar_for_gene
        out = run_gistar_for_gene(args.gene, args.gsea_dir, args.position_cellline_csv, args.nmf_root, args.output_root, not args.no_gzip)
    elif args.cmd == "tahoe-confidence":
        from starmap.stages.tahoe import run_tahoe_confidence
        out = run_tahoe_confidence(args.gene, args.sorted_logodds_csv, args.perdrug_dir, args.output_dir, args.pvalue_cutoff)
    elif args.cmd in legacy:
        run_legacy(legacy[args.cmd], args.argument)
        return 0
    else:
        parser.error(f"Unknown command: {args.cmd}")
        return 2
    print(f"Saved {out}" if out else "No output generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
