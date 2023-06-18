import os
import pandas as pd


def preprocess_english_simlex999(path):
    """
    Preprocess the simlex999 dataset and produce a file which fits the schema (word1, word2, similarity).
    Args:
        path - the path of the SimLex999.txt file (download at https://fh295.github.io/simlex.html)

    Returns:
        the path to the preprocessed file. The preprocessed file is a tsv where lines are 3-tuples, each consisting of a word pair and a similarity value.
        Example: `word1    word2  similarity`.
    """
    file_dir, file_name = os.path.dirname(path), os.path.basename(path)
    output_path = os.path.join(file_dir, f"preprocessed-{file_name}")

    df = pd.read_csv(path, sep="\t")
    df.rename(columns={"SimLex999": "similarity"}, inplace=True)
    out_df = df[["word1", "word2", "similarity"]]
    out_df.to_csv(output_path, sep="\t", index=False)


preprocess_english_simlex999("../data/input/English_testset/SimLex-999.txt")
