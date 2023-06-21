from typing import List, Dict, Callable
import pyarabic.araby as araby


def remove_niqqud_from_string(s: str) -> str:
    """Hebrew may have the niqqud diacritics. We want to remove them for consistency."""
    return "".join(["" if 1456 <= ord(c) <= 1479 else c for c in s])


LANG_TO_NORMALIZE_FCT: Dict[str, Callable[[str], str]] = {
    "English": (lambda s: s),
    "Spanish": (lambda s: s),
    "German": (lambda s: s),
    "Polish": (lambda s: s),
    "Arabic": araby.strip_diacritics,
    "Hebrew": remove_niqqud_from_string,
}


def warn_if_no_normalization_fct(lang: str):
    if lang not in LANG_TO_NORMALIZE_FCT.keys():
        print(
            f"WARNING: language {lang} does not have an explicit normalization function defined. Please implement the normalization function in the LANG_TO_NORMALIZE_FCT dictionary."
        )


def normalize_words(word_list: List[str], lang: str) -> List[str]:
    warn_if_no_normalization_fct(lang)

    # Need to remove duplicates because sometimes normalizing a word results in an existing word.
    return list(set([LANG_TO_NORMALIZE_FCT[lang](word) for word in word_list]))


def normalize_word(word: str, lang: str) -> List[str]:
    warn_if_no_normalization_fct(lang)

    # Need to remove duplicates because sometimes normalizing a word results in an existing word.
    return LANG_TO_NORMALIZE_FCT[lang](word)


def normalize_eval_set_word_pairs(df, lang):
    """
    Given a df with columns ["word1", "word2", "similarity"], normalize all words in cols `word1` and `word2` according to lang and average similarity of rows that consequently contain duplicate pairs.
    """
    df["word1"] = df["word1"].apply(lambda s: normalize_word(s, lang))
    df["word2"] = df["word2"].apply(lambda s: normalize_word(s, lang))
    agg_df = df.groupby(["word1", "word2"]).agg(["mean", "max", "min"]).reset_index()

    # Sanity check that aggregated word pairs from normalization don't differ too much in similarity judgment
    agg_df["percent_range_diff"] = (agg_df["similarity"]["max"] - agg_df["similarity"]["min"]) / max(df["similarity"])
    print(
        "The following word pairs have a differing max and min similarity rating:",
        agg_df[agg_df["percent_range_diff"] != 0].sort_values(by="percent_range_diff", ascending=False),
    )

    agg_df.columns = agg_df.columns.map("".join)
    out_df = agg_df[["word1", "word2", "similaritymean"]]
    out_df.rename(
        columns={
            "similaritymean": "similarity",
        },
        inplace=True,
    )
    return out_df
