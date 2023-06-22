import os
import pandas as pd

from utils import normalize_eval_set_word_pairs


def preprocess_portuguese_simlex999(path, lang):
    """
    Preprocess the LX-simlex999 dataset (portuguese) and produce a file which fits the schema (word1, word2, similarity).
    Args:
        path - the path of the LX-SimLex999.txt file (download at https://portulanclarin.net/repository/download/4ab1ea58e6d311e6a2aa782bcb0741351e920e18429e4d3e9d229a58030812fe/)

    Returns:
        the path to the preprocessed file. The preprocessed file is a tsv where lines are 3-tuples, each consisting of a word pair and a similarity value.
        Example: `word1    word2  similarity`.
    """
    file_dir, file_name = os.path.dirname(path), os.path.basename(path)
    output_path = os.path.join(file_dir, f"preprocessed-{file_name}")

    df = pd.read_csv(path, sep="\t")
    df.rename(columns={"LX-SimLex-999": "similarity"}, inplace=True)
    out_df = df[["word1", "word2", "similarity"]]

    out_df = normalize_eval_set_word_pairs(out_df, lang=lang)
    out_df.to_csv(output_path, sep="\t", index=False)


preprocess_portuguese_simlex999("../data/input/Portuguese_testset/LX-SimLex-999.txt", lang="Portuguese")
