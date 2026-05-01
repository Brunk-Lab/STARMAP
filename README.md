![STARMAP](figures/starmaplogo.png)

# STARMAP Pipeline: Structure-based Topological Analysis of Regulatory and Molecular Activity Patterns

Installable Python package for the STARMAP analysis workflow. Please note that some analyses from the paper are highly dependent on folder structure, and are included as legacy scripts for users to run and modify on their own.

The original scripts are preserved in `src/starmap/legacy/` for reference. The main package code lives in `src/starmap/stages/`.

## License

This project is released under a custom restricted use license.

### Summary
- Free for non-commercial academic and research use  
- Modification and redistribution allowed for academic purposes  
- Commercial use is strictly prohibited without explicit permission  

For commercial licensing or other inquiries, please contact elizabeth_brunk@med.unc.edu.


## Install

From the repository root:

```bash
pip install -e .
```

For the full workflow, including MLP and Gi* dependencies:

```bash
pip install -e ".[all]"
```

For Longleaf, install inside your conda environment.

## Pipeline overview

```text
3D coordinates
  -> NMF flatmap and clusters
  -> Gi* TRN scoring
  -> AlphaFold distance features
  -> CTRP sensitivity and cluster annotation
  -> MLP drug response models
  -> cluster-level drug log2 odds ratios
  -> Tahoe/Perturb-seq empirical confidence
```

## Command-line usage

### 1. NMF flatmap

```bash
starmap nmf TP53 \
  --coord-csv data/3Dcoord_allgenes.csv \
  --output-dir output/nmf
```

Output:

```text
output/nmf/T/TP53_nmfinfo_final.csv
```

Function:

```python
from starmap.stages.nmf import run_nmf_for_gene

run_nmf_for_gene(
    gene="TP53",
    coord_csv="data/3Dcoord_allgenes.csv",
    output_dir="output/nmf",
    min_components=3,
    max_components=6,
    random_state=0,
)
```

Expected coordinate columns:

```text
gene, x_coord, y_coord, z_coord, res
```

Returned output is the saved CSV path.

### 2. AlphaFold distance features

```bash
starmap distances P04637 \
  --input-csv data/ParsedMutations_Plus_Sites_Filtered.csv \
  --output-dir output/dist_files
```

Use a local PDB instead of downloading AlphaFold:

```bash
starmap distances P04637 \
  --input-csv data/ParsedMutations_Plus_Sites_Filtered.csv \
  --output-dir output/dist_files \
  --pdb-path data/P04637.pdb
```

Output:

```text
output/dist_files/P04637/P04637_Distances.csv
```

Function:

```python
from starmap.stages.distances import run_distance_generation

run_distance_generation(
    uniprot_id="P04637",
    input_csv="data/ParsedMutations_Plus_Sites_Filtered.csv",
    output_dir="output/dist_files",
    alphafold_id_col="AlphaFold_IDs",
    pdb_path=None,
)
```

Expected mutation columns include:

```text
AlphaFold_IDs, AA_POS, ID, CELL_LINE, SIFT, LIKELY_LOF, protein_change,
ActiveSitePositions, BindingSites
```

### 3. Add sensitivity labels and cluster IDs

```bash
starmap annotate P04637 \
  --ccle-csv data/CCLE_cell_line_drugsensitivity_10072023.csv \
  --dist-dir output/dist_files \
  --map-csv data/gene_uniprot_res_clust.csv
```

Output:

```text
output/dist_files/P04637/P04637_Distances_Labeled.parquet
```

Function:

```python
from starmap.stages.annotate import run_annotate_auc_cluster

run_annotate_auc_cluster(
    uniprot_id="P04637",
    ccle_csv="data/CCLE_cell_line_drugsensitivity_10072023.csv",
    dist_dir="output/dist_files",
    map_csv="data/gene_uniprot_res_clust.csv",
    output_suffix="_Distances_Labeled.parquet",
    sensitivity_quantile=0.25,
)
```

Expected CCLE columns:

```text
cell_line, drug, AUC_CTRP
```

Expected cluster-map columns can use common variants of:

```text
uniprot_ID or uniprot_id, res or residue_position, clust or cluster_id
```

### 4. Train MLP drug-response models

```bash
starmap mlp P04637 \
  --dist-root output/dist_files \
  --output-root output/MLP_outputs
```

Run one drug only:

```bash
starmap mlp P04637 \
  --dist-root output/dist_files \
  --output-root output/MLP_outputs \
  --drug paclitaxel
```

Output per protein and drug:

```text
output/MLP_outputs/P04637/<drug>/mlp_predictions.csv
output/MLP_outputs/P04637/<drug>/mlp_model.joblib
output/MLP_outputs/P04637/<drug>/mlp_report.txt
```

Function:

```python
from starmap.stages.mlp import run_mlp_for_uniprot

run_mlp_for_uniprot(
    uniprot_id="P04637",
    dist_root="output/dist_files",
    output_root="output/MLP_outputs",
    drug=None,
    use_clust=True,
    max_epochs=200,
)
```

Input is the labeled distance table from the annotation stage.

### 5. Cluster-level drug log-odds

```bash
starmap logodds P04637 \
  --mlp-root output/MLP_outputs \
  --nmf-root output/nmf \
  --mapping-csv data/uniprot_gene_map.csv \
  --output-root output/logodds_results
```

Outputs:

```text
output/logodds_results/TP53/TP53_<drug>_logodds.csv
output/logodds_results/TP53/sorted_mlp.csv
```

Function:

```python
from starmap.stages.logodds import run_logodds_for_uniprot

run_logodds_for_uniprot(
    uniprot_id="P04637",
    mlp_root="output/MLP_outputs",
    nmf_root="output/nmf",
    mapping_csv="data/uniprot_gene_map.csv",
    output_root="output/logodds_results",
    models=["mlp"],
)
```

Expected mapping columns:

```text
uniprot_id, gene
```

### 6. Gi* TRN scoring

```bash
starmap gistar TP53 \
  --gsea-dir data/GSEA_files \
  --position-cellline-csv data/ccle_gene_position_cellline.csv \
  --nmf-root output/nmf \
  --output-root output/gistar
```

Outputs:

```text
output/gistar/scores/T/TP53_scores_all_trns.csv
output/gistar/3D_files/T/TP53_<TRN>_gdf.csv.gz
```

Function:

```python
from starmap.stages.gistar import run_gistar_for_gene

run_gistar_for_gene(
    gene="TP53",
    gsea_dir="data/GSEA_files",
    position_cellline_csv="data/ccle_gene_position_cellline.csv",
    nmf_root="output/nmf",
    output_root="output/gistar",
)
```

### 7. Tahoe empirical confidence

```bash
starmap tahoe-confidence TP53 \
  --sorted-logodds-csv output/logodds_results/TP53/sorted_mlp.csv \
  --perdrug-dir output/TP53_mut_v_other_perdrug \
  --output-dir output/emp_files
```

Output:

```text
output/emp_files/TP53_emp.csv
```

Function:

```python
from starmap.stages.tahoe import run_tahoe_confidence

run_tahoe_confidence(
    gene="TP53",
    sorted_logodds_csv="output/logodds_results/TP53/sorted_mlp.csv",
    perdrug_dir="output/TP53_mut_v_other_perdrug",
    output_dir="output/emp_files",
)
```

## Legacy commands

Some Perturb-seq and Tahoe expression scripts are still available as legacy wrappers because they depend strongly on large local file layout assumptions:

```bash
starmap download-perturbseq TP53
starmap preprocess-perturbseq TP53
starmap perturbseq-gene-output TP53
starmap perturbseq-confidence TP53
starmap tahoe-expression-legacy TP53
```

Use these only if your directory structure matches the original scripts.

## Recommended repository layout

```text
project/
  data/
  output/
  src/starmap/
  examples/
  slurm/
  pyproject.toml
  README.md
```

