# Import necessary packages
import pandas as pd
import gseapy as gp
import os
import sys

# Gene name from command line
# Check input
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py GENE")
    sys.exit(1)

input_gene = sys.argv[1].upper().strip()

# Config
files_dir = os.listdir('../data/tahoe_data_pseudobulk_cellline')
drugs = [item.replace('_pseudobulk.parquet', '') for item in files_dir]
if '.ipynb_checkpoints' in drugs:
    drugs.remove('.ipynb_checkpoints')

mut = pd.read_csv('../data/OmicsSomaticMutations.csv')
info = pd.read_csv('../data/sample_info.csv')

# Create mapping from RRID (DepMap_ID) → stripped cell line name
rrid_to_name = dict(zip(info["DepMap_ID"], info["CCLE_Name"]))

# Add a new column to `mut` using its existing ModelID column
mut["CCLE_Name"] = mut["ModelID"].map(rrid_to_name)

sub_mut = [*set(mut[mut['HugoSymbol']==input_gene]['CCLE_Name'].tolist())]

os.makedirs(f'../output/{input_gene}_mut_v_other_perdrug',exist_ok=True)

for drug in drugs:
    df = pd.read_parquet(f'../data/tahoe_data_pseudobulk_cellline/{drug}_pseudobulk.parquet')
    # Create a mapping from RRID → stripped_cell_line_name
    rrid_to_name = dict(zip(info["RRID"], info["CCLE_Name"]))

    # Rename df’s columns using this mapping (ignore missing ones)
    df = df.rename(columns=rrid_to_name)

    cell_lines = df.columns.tolist()
    clust = [i for i in cell_lines if i in sub_mut]
    other = [i for i in cell_lines if i not in sub_mut]

    clustdf = df[clust]
    otherdf = df[other]

    trn_dir = '../../5_perturbseq_xatlas/data/trn_gene_sets'
    trn_dir_in = os.listdir(trn_dir)
    trns = [item.replace('_geneset.csv', '') for item in trn_dir_in]
    if ".ipynb_checkpoints" in trns:
        trns.remove(".ipynb_checkpoints")

    clust_all = []
    other_all = []

    for trn in trns:
        # Load gene list
        geneset_path = os.path.join(trn_dir, f'{trn}_geneset.csv')
        genelist = pd.read_csv(geneset_path)['0'].tolist()

        # Run ssGSEA for cluster cells
        ssgsea_clust = gp.ssgsea(
            data=clustdf,
            gene_sets={"TARGETS": genelist},
            sample_norm_method="rank",
            outdir=None,
            min_size=1,
            max_size=2000,
            verbose=False
        )
        clust_scores = ssgsea_clust.res2d  # rows=cells, cols=gene set
        clust_scores.rename(columns={'NES': trn}, inplace=True)
        clust_all.append(pd.DataFrame(clust_scores[trn]))

        # Run ssGSEA for other cells
        ssgsea_other = gp.ssgsea(
            data=otherdf,
            gene_sets={"TARGETS": genelist},
            sample_norm_method="rank",
            outdir=None,
            min_size=1,
            max_size=2000,
            verbose=False
        )
        other_scores = ssgsea_other.res2d
        other_scores.rename(columns={'NES': trn}, inplace=True)
        other_all.append(pd.DataFrame(other_scores[trn]))

    # ---- Combine across all TRNs ----
    clust_df_all = pd.concat(clust_all, axis=1)
    other_df_all = pd.concat(other_all, axis=1)

    # ---- Save if needed ----
    clust_df_all.to_csv(f"../output/{input_gene}_mut_v_other_perdrug/ssgsea_cluster_all_trns_{drug}.csv")
    other_df_all.to_csv(f"../output/{input_gene}_mut_v_other_perdrug/ssgsea_other_all_trns_{drug}.csv")