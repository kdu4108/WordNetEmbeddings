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
    "Portuguese": (lambda s: s),
    "Arabic": araby.strip_diacritics,
    "Hebrew": remove_niqqud_from_string,
}


def warn_if_no_normalization_fct(lang: str):
    if lang not in LANG_TO_NORMALIZE_FCT.keys():
        print(
            f"WARNING: language {lang} does not have an explicit normalization function defined. Please implement the normalization function in the LANG_TO_NORMALIZE_FCT dictionary."
        )


def normalize_words(word_list: List[str], lang: str) -> List[str]:
    """Normalize a a list of words for given language according to LANG_TO_NORMALIZE_FCT. For example, this is useful for standardizing away diacritics in Hebrew and Arabic."""
    warn_if_no_normalization_fct(lang)

    # Need to remove duplicates because sometimes normalizing a word results in an existing word.
    return list(set([LANG_TO_NORMALIZE_FCT[lang](word) for word in word_list]))


def normalize_word(word: str, lang: str) -> List[str]:
    """Normalize a word for given language according to LANG_TO_NORMALIZE_FCT. For example, this is useful for standardizing away diacritics in Hebrew and Arabic."""
    warn_if_no_normalization_fct(lang)

    # Need to remove duplicates because sometimes normalizing a word results in an existing word.
    return LANG_TO_NORMALIZE_FCT[lang](word)
