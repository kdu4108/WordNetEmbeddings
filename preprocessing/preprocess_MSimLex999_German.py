import os
import pandas as pd


def preprocess_german_simlex999(path):
    """
    Preprocess the German simlex999 dataset (downloaded from https://leviants.com/multilingual-simlex999-and-wordsim353/) and produce a file which fits the schema (word1, word2, similarity).
    Args:
        path - the path of the MSimLex999_German.txt file

    Returns:
        the path to the preprocessed file. The preprocessed file is a tsv where lines are 3-tuples, each consisting of a word pair and a similarity value.
        Example: `word1    word2  similarity`.
    """
    file_dir, file_name = os.path.dirname(path), os.path.basename(path)
    output_path = os.path.join(file_dir, f"preprocessed-{file_name}")

    df = pd.read_csv(path)

    out_df = df[["Word1", "Word2", "Average Score"]]
    out_df.rename(
        columns={
            "Word1": "word1",
            "Word2": "word2",
            "Average Score": "similarity",
        },
        inplace=True,
    )
    out_df.to_csv(output_path, sep="\t", index=False)


preprocess_german_simlex999("../data/input/German_testset/MSimLex999_German.txt")
