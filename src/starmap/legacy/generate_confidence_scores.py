# Import necessary packages
import sys
import ast
from pathlib import Path
import pandas as pd

# Set up paths
DF_PATH = f"../output/{gene}_perturbed_df.csv"
BASE_SCORE_DIR = Path("../../2_gistar/output/scores")

# Functions

def load_main_df(path: str) -> pd.DataFrame:
    """Load main perturbed dataframe and clean pathway names."""
    df = pd.read_csv(path)
    df["pathway"] = df["pathway"].str.replace("_geneset", "", regex=False)
    return df


def load_score_csv(score_path: Path) -> pd.DataFrame | None:
    """Load and expand the scores CSV for a given gene."""
    if not score_path.exists():
        print(f"Score file not found: {score_path}")
        return None

    try:
        score = pd.read_csv(score_path)
        score["scores"] = score["scores"].apply(ast.literal_eval)

        scores_expanded = pd.DataFrame(score["scores"].tolist(), index=score.index)
        scores_expanded.columns = [f"score_{i+1}" for i in range(scores_expanded.shape[1])]

        score = pd.concat([score.drop(columns=["scores"]), scores_expanded], axis=1)
        score["pathway"] = score["pathway_file"].str.replace("_GSEA.csv", "", regex=False)

        score_cols = [c for c in score.columns if c.startswith("score_")]
        score["score_max"] = score[score_cols].max(axis=1)

        return score

    except Exception as e:
        print(f"Error reading {score_path.name}: {e}")
        return None


def build_sub(df: pd.DataFrame, gene: str, score: pd.DataFrame) -> pd.DataFrame | None:
    """Generate the sub dataframe for one gene based on score_max ranking."""
    try:
        if score is None or score["score_max"].mean() == 0:
            print(f"No usable score data for {gene}")
            return None

        sub = df[df["gene"].str.upper() == gene.upper()].copy()
        if sub.empty:
            print(f"No rows found in main dataframe for gene {gene}")
            return None

        ranked = score.sort_values(by="score_max", ascending=False)

        # keep only pathways present in both dataframes, in ranked order
        common_pathways = ranked["pathway"][ranked["pathway"].isin(sub["pathway"])]
        sub = sub.set_index("pathway").loc[common_pathways].reset_index()

        sub = sub.reset_index().rename(columns={"index": "adjusted_rank"})
        sub["adjusted_rank"] += 1
        sub["significant"] = sub["pval"] < 0.05
        sub["cum_significant"] = sub["significant"].cumsum()
        sub["confidence"] = sub["cum_significant"] / sub["adjusted_rank"]
        sub = sub[sub["confidence"] != 0].copy()

        if not sub.empty:
            min_val = sub["confidence"].min()
            max_val = sub["confidence"].max()

            if max_val > min_val:
                # normalize to 0-1
                sub["confidence"] = (sub["confidence"] - min_val) / (max_val - min_val)

                # dynamic linear rescale based on first adjusted rank
                rank_anchor = sub["adjusted_rank"].iloc[0]
                max_rank = 500
                rank_norm = min(rank_anchor / max_rank, 1.0)

                start_val = 1 - rank_norm
                sub["confidence"] = sub["confidence"] * start_val
            else:
                sub["confidence"] = 0

        sub["gene"] = gene
        return sub

    except Exception as e:
        print(f"Error building sub for {gene}: {e}")
        return None


# Run
if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Please enter in the format: python script.py GENE")
        sys.exit(1)

    gene = sys.argv[1].upper().strip()
    out_path = Path(f"../output/{gene}_sub.csv")

    df = load_main_df(DF_PATH)

    letter = gene[0].upper()
    score_path = BASE_SCORE_DIR / letter / f"{gene}_scores_all_trns.csv"

    score = load_score_csv(score_path)
    sub = build_sub(df, gene, score)

    if sub is not None and not sub.empty:
        sub.to_csv(out_path, index=False)
        print(sub.head())
        print(f"\nSaved {len(sub)} rows to {out_path}")
    else:
        print(f"No sub dataframe generated for {gene}")