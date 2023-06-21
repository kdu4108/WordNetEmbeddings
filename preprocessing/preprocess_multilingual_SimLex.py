import os
import pandas as pd
from typing import List

from utils import normalize_eval_set_word_pairs


def preprocess_multilingual_simlex(path, lang):
    """
    Preprocess a language within the Multilingual Simlex dataset (https://direct.mit.edu/coli/article/46/4/847/97326/Multi-SimLex-A-Large-Scale-Evaluation-of) and produce a file which fits the schema (word1, word2, similarity).
    Args:
        path - the path of the {LANG}-SimLex.csv file (download at https://multisimlex.com/#download)

    Returns:
        the path to the preprocessed file. The preprocessed file is a tsv where lines are 3-tuples, each consisting of a word pair and a similarity value.
        Example: `word1    word2  similarity`.
    """
    file_dir, file_name = os.path.dirname(path), os.path.basename(path)
    output_path = os.path.join(file_dir, f"preprocessed-{file_name}")

    df = pd.read_csv(path)
    annotator_cols: List[str] = [col for col in df.columns if col.startswith("Annotator")]
    mean_similarity_scores: pd.Series = df[annotator_cols].mean(
        axis=1
    )  # average across all annotators for each pair of words

    out_df = df[["Word 1", "Word 2"]]
    out_df["similarity"] = mean_similarity_scores
    out_df.rename(
        columns={
            "Word 1": "word1",
            "Word 2": "word2",
        },
        inplace=True,
    )

    out_df = normalize_eval_set_word_pairs(out_df, lang=lang)
    out_df.to_csv(output_path, sep="\t", index=False)


# Needed because we choose to keep the abbreviations from the files downloaded from https://multisimlex.com/#download consistent-ish.
# This way, there's provenance connecting the filename from the download to the language we're using.
lang_to_abbrev = {
    "Arabic": "ARA",
    "Polish": "POL",
    "Hebrew": "HEB",
    "Spanish": "SPA",
}
for lang, abbrev in lang_to_abbrev.items():
    preprocess_multilingual_simlex(f"../data/input/{lang}_testset/{abbrev}-SimLex.csv", lang)
