import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from scipy import stats
import re
import os
import sys

# Gene name from command line
# Check input
if len(sys.argv) < 2 or not sys.argv[1].strip():
    print("Please enter in the format: python script.py GENE")
    sys.exit(1)

input_gene = sys.argv[1].upper().strip()

# Define functions
def normalize_drug_name(name):
    """Normalize drug names to a comparable lowercase stripped form."""
    name = name.lower().strip()
    name = re.sub(r'^(random|positive|negative)_', '', name)     # remove prefixes
    name = re.sub(r'\(.*?\)', '', name)                          # remove parentheses
    name = re.sub(r'[^a-z0-9\-]+', '', name)                     # remove non-alphanumeric chars
    return name

df = pd.read_csv(f'../../4_drug_logodds/output/logodds_results/{input_gene}/sorted_mlp.csv')

control_clust = pd.read_csv(f'../output/{input_gene}_mut_v_other_perdrug/ssgsea_cluster_all_trns_DMSO_TF.csv')
cols_to_use = [c for c in control_clust.columns[:10] if c != "Unnamed: 0"]
control_clust["mean_expr"] = control_clust[cols_to_use].mean(axis=1)

list = [f for f in os.listdir(f'../output/{input_gene}_mut_v_other_perdrug') if f.endswith(".csv")]
drugs = [item.replace('ssgsea_cluster_all_trns_', '') for item in list]
drugs = [item.replace('ssgsea_other_all_trns_', '') for item in drugs]
drugs = [item.replace('.csv', '') for item in drugs]
drugs = [*set(drugs)]

clust_testvcontrol = []
test_clustvother = []

for drug in drugs:
    # --- Load cluster file ---
    test_clust = pd.read_csv(f'../output/{input_gene}_mut_v_other_perdrug/ssgsea_cluster_all_trns_{drug}.csv')
    cols_to_use_clust = [c for c in test_clust.columns[:10] if c != "Unnamed: 0"]
    test_clust["mean_expr"] = test_clust[cols_to_use_clust].mean(axis=1)

    # --- Load "other" file ---
    test_other = pd.read_csv(f'../output/{input_gene}_mut_v_other_perdrug/ssgsea_other_all_trns_{drug}.csv')
    cols_to_use_other = [c for c in test_other.columns[:10] if c != "Unnamed: 0"]
    test_other["mean_expr"] = test_other[cols_to_use_other].mean(axis=1)

    # --- Rank-sum tests ---
    stat1, pval1 = ranksums(test_clust["mean_expr"], control_clust["mean_expr"])
    stat2, pval2 = ranksums(test_clust["mean_expr"], test_other["mean_expr"])

    clust_testvcontrol.append(pval1)
    test_clustvother.append(pval2)

stat = pd.DataFrame({'Drug': drugs, 'clust_testvcontrol': clust_testvcontrol, 'test_clustvother': test_clustvother})

# Apply normalization
stat["norm_drug"] = stat["Drug"].apply(normalize_drug_name)
df["norm_drug"] = df["drug"].apply(normalize_drug_name)

df = df.reset_index()
df['index'] = df['index']+1

merged = pd.merge(df, stat, on="norm_drug", how="inner")
merged = merged.drop_duplicates().reset_index(drop=True)
series = merged['index']
merged = merged.sort_values(by=['log2_odds_ratio','clust_testvcontrol'],ascending=[False, True]).reset_index(drop=True)
merged['index']=series
merged = merged.reset_index()
merged['rank'] = merged['level_0'] + 1
sub = merged[['index','rank','clust_testvcontrol','test_clustvother']].copy()
sub = sub.sort_values('rank')
sub['significant'] = sub['clust_testvcontrol'] < 0.05

sub['cum_significant'] = sub['significant'].cumsum()
sub['confidence'] = sub['cum_significant'] / sub['rank']

emp = sub[['index','confidence']].copy()
emp['norm_confidence'] = (emp['confidence'] - emp['confidence'].min()) / (emp['confidence'].max() - emp['confidence'].min())
emp['protein']=input_gene

os.makedirs('../output/emp_files',exist_ok=True)

emp.to_csv(f'../output/emp_files/{input_gene}_emp.csv')