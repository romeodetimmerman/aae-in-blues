import pandas as pd
import numpy as np
import re
from g2p_en import G2p

g2p = G2p()

vowels = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
    "AX",
}

nasals = {"M", "N", "NG"}

liquids = {"L", "R"}

glides = {"W", "Y", "HH"}

obstruents = {
    "P",
    "B",
    "T",
    "D",
    "K",
    "G",
    "F",
    "V",
    "TH",
    "DH",
    "S",
    "Z",
    "SH",
    "ZH",
    "CH",
    "JH",
}

bilabials = {"P", "B", "M"}
labiodentals = {"F", "V"}
dentals = {"TH", "DH"}
alveolars = {"T", "D", "S", "Z", "N", "L", "R"}
postalveolars = {"SH", "ZH", "CH", "JH"}
palatals = {"Y"}
velars = {"K", "G", "NG"}
glottals = {"HH"}
labiovelars = {"W"}


def strip_stress(phoneme: str) -> str:
    """
    remove primary/secondary stress digits from an arpabet phoneme

    params
    ------
    phoneme: str
        arpabet phoneme

    returns
    -------
    phoneme: str
        arpabet phoneme without stress digits
    """
    return re.sub(r"\d", "", phoneme).upper()


def classify_phoneme(p: str) -> str:
    """
    map a single arpabet phoneme (without stress) to:

    params
    ------
    p: str
        phoneme in arpabet without stress digits

    returns
    -------
    class_label: str
        vowel, nasal, liquid, glide, obstruent, or other
    """
    p_clean = strip_stress(p)
    if p_clean in vowels:
        return "vowel"
    if p_clean in nasals:
        return "nasal"
    if p_clean in liquids:
        return "liquid"
    if p_clean in glides:
        return "glide"
    if p_clean in obstruents:
        return "obstruent"
    return "other"


def classify_place(p: str) -> str:
    """
    map a single arpabet phoneme (without stress) to place of articulation

    params
    ------
    p: str
        phoneme in arpabet without stress digits

    returns
    -------
    place_label: str
        bilabial, labiodental, dental, alveolar, postalveolar, palatal,
        velar, glottal, labiovelar, vowel, or other
    """
    p_clean = strip_stress(p)
    if p_clean in vowels:
        return "vowel"
    if p_clean in bilabials:
        return "bilabial"
    if p_clean in labiodentals:
        return "labiodental"
    if p_clean in dentals:
        return "dental"
    if p_clean in alveolars:
        return "alveolar"
    if p_clean in postalveolars:
        return "postalveolar"
    if p_clean in palatals:
        return "palatal"
    if p_clean in velars:
        return "velar"
    if p_clean in glottals:
        return "glottal"
    if p_clean in labiovelars:
        return "labiovelar"
    return "other"


def word_to_phonemes(word: str) -> list[str]:
    """
    convert word to list of arpabet phonemes without stress digits

    params
    ------
    word: str
        target word

    returns
    -------
    phonemes: list[str]
        list of arpabet phonemes without stress digits
    """
    if not isinstance(word, str) or not word.strip():
        return []
    phonemes = []
    for token in g2p(word):
        cleaned = strip_stress(token)
        if cleaned.isalpha():
            phonemes.append(cleaned)
    return phonemes


def normalize_feature(value: str) -> str:
    """
    normalize aae feature label for consistent comparisons

    params
    ------
    value: str
        raw aae feature

    returns
    -------
    value: str
        normalized aae feature label
    """
    if not isinstance(value, str):
        return ""
    return value.strip().lower().replace("’", "'")


def anchor_phoneme_index(phonemes: list[str], feature: str) -> int | None:
    """
    select the phoneme index inside the target word that anchors the feature
    """
    normalized = normalize_feature(feature)
    if not phonemes:
        return None
    if normalized == "ai monophthongization":
        for idx, p in enumerate(phonemes):
            if strip_stress(p) == "AY":
                return idx
        return None
    if normalized == "ing ultimas":
        return len(phonemes) - 1
    if normalized == "post-consonantal d":
        if strip_stress(phonemes[-1]) == "D":
            if len(phonemes) >= 2 and classify_phoneme(phonemes[-2]) != "vowel":
                return len(phonemes) - 1
        return None
    if normalized == "post-consonantal t":
        if strip_stress(phonemes[-1]) == "T":
            if len(phonemes) >= 2 and classify_phoneme(phonemes[-2]) != "vowel":
                return len(phonemes) - 1
        return None
    if normalized == "post-vocalic r":
        final_clean = strip_stress(phonemes[-1])
        if final_clean == "ER":
            return len(phonemes) - 1
        if final_clean == "R":
            if len(phonemes) >= 2 and classify_phoneme(phonemes[-2]) == "vowel":
                return len(phonemes) - 1
        return None
    if normalized == "third person singular":
        if strip_stress(phonemes[-1]) in {"S", "Z"}:
            return len(phonemes) - 1
        return None
    if normalized in {"auxiliary verb", "zero copula"}:
        return None
    return None


def segment_label(manner: str):
    """
    map phoneme manner to vowel/consonant
    """
    if pd.isna(manner):
        return np.nan
    return "vowel" if manner == "vowel" else "consonant"


def classify_optional(phoneme: str | None, classifier):
    """
    safely classify a phoneme that may be missing
    """
    if phoneme is None:
        return np.nan
    return classifier(phoneme)


def extract_phoneme_context(row: pd.Series) -> pd.Series:
    """
    derive contextual phonology around the feature anchor in the target word

    params
    ------
    row: pd.Series
        row with word, previous_word, next_word, aae_feature

    returns
    -------
    context: pd.Series
        manners, places, and segments for previous/next phonemes
    """
    phonemes = word_to_phonemes(row.get("word", ""))
    feature_value = row.get("aae_feature", "")
    normalized_feature = normalize_feature(feature_value)
    anchor_idx = anchor_phoneme_index(phonemes, feature_value)
    if anchor_idx is None:
        return pd.Series(
            {
                "prev_manner": np.nan,
                "next_manner": np.nan,
                "prev_place": np.nan,
                "next_place": np.nan,
                "prev_segment": np.nan,
                "next_segment": np.nan,
            }
        )
    prev_phoneme = phonemes[anchor_idx - 1] if anchor_idx - 1 >= 0 else None
    next_phoneme = phonemes[anchor_idx + 1] if anchor_idx + 1 < len(phonemes) else None
    if normalized_feature == "post-vocalic r":
        if anchor_idx >= 0:
            for idx in range(anchor_idx - 1, -1, -1):
                if classify_phoneme(phonemes[idx]) == "vowel":
                    prev_phoneme = phonemes[idx]
                    break
            else:
                prev_phoneme = None
    if prev_phoneme is None:
        fallback_prev = word_to_phonemes(row.get("previous_word", ""))
        if fallback_prev:
            prev_phoneme = fallback_prev[-1]
    if next_phoneme is None:
        fallback_next = word_to_phonemes(row.get("next_word", ""))
        if fallback_next:
            next_phoneme = fallback_next[0]
    prev_manner = classify_optional(prev_phoneme, classify_phoneme)
    next_manner = classify_optional(next_phoneme, classify_phoneme)
    prev_place = classify_optional(prev_phoneme, classify_place)
    next_place = classify_optional(next_phoneme, classify_place)
    prev_segment = segment_label(prev_manner)
    next_segment = segment_label(next_manner)
    return pd.Series(
        {
            "prev_manner": prev_manner,
            "next_manner": next_manner,
            "prev_place": prev_place,
            "next_place": next_place,
            "prev_segment": prev_segment,
            "next_segment": next_segment,
        }
    )


def previous_phoneme_manner(word: str):
    """
    return the class of the final phoneme of the word

    params
    ------
    word: str
        target word

    returns
    -------
    class_label: str
        np.nan if unavailable, vowel, nasal, liquid, glide, obstruent, other
    """
    phonemes = word_to_phonemes(word)
    if not phonemes:
        return np.nan
    return classify_phoneme(phonemes[-1])


def next_phoneme_manner(word: str):
    """
    return the class of the initial phoneme of the word:

    params
    ------
    word: str
        target word

    returns
    -------
    class_label: str
        np.nan if unavailable, vowel, nasal, liquid, glide, obstruent, other
    """
    phonemes = word_to_phonemes(word)
    if not phonemes:
        return np.nan
    return classify_phoneme(phonemes[0])


def previous_phoneme_place(word: str):
    """
    return place of articulation for final phoneme of previous word

    params
    ------
    word: str
        target word

    returns
    -------
    place_label: str
        np.nan if unavailable, bilabial, labiodental, dental, alveolar,
        postalveolar, palatal, velar, glottal, labiovelar, vowel, other
    """
    phonemes = word_to_phonemes(word)
    if not phonemes:
        return np.nan
    return classify_place(phonemes[-1])


def next_phoneme_place(word: str):
    """
    return place of articulation for initial phoneme of next word

    params
    ------
    word: str
        target word

    returns
    -------
    place_label: str
        np.nan if unavailable, bilabial, labiodental, dental, alveolar,
        postalveolar, palatal, velar, glottal, labiovelar, vowel, other
    """
    phonemes = word_to_phonemes(word)
    if not phonemes:
        return np.nan
    return classify_place(phonemes[0])


def previous_segment(word: str):
    """
    return vowel/consonant for final phoneme of previous word

    params
    ------
    word: str
        target word

    returns
    -------
    segment: str
        np.nan if unavailable, vowel, consonant
    """
    label = previous_phoneme_manner(word)
    if pd.isna(label):
        return np.nan
    return "vowel" if label == "vowel" else "consonant"


def next_segment(word: str):
    """
    return vowel/consonant for initial phoneme of next word

    params
    ------
    word: str
        target word

    returns
    -------
    segment: str
        np.nan if unavailable, vowel, consonant
    """
    label = next_phoneme_manner(word)
    if pd.isna(label):
        return np.nan
    return "vowel" if label == "vowel" else "consonant"


############################################
# morphological status of the final stop   #
############################################

TD_FEATURES = {"post-consonantal t", "post-consonantal d"}
NOT_APPLICABLE = "not_applicable"

# stems that precede -n't, used to catch negative contractions
NEG_STEMS = {
    "don",
    "ain",
    "can",
    "won",
    "didn",
    "wouldn",
    "wasn",
    "weren",
    "couldn",
    "shouldn",
    "haven",
    "hasn",
    "hadn",
    "doesn",
    "isn",
    "aren",
    "mustn",
    "needn",
    "mightn",
}

# high-frequency function word
FUNCTION_WORDS = {"and"}

# irregular "semiweak" pasts
SEMIWEAK = {
    "kept",
    "left",
    "lost",
    "meant",
    "felt",
    "swept",
    "went",
    "sent",
    "spent",
    "bent",
    "lent",
    "dealt",
    "dreamt",
    "wept",
    "crept",
    "slept",
    "built",
    "burnt",
    "spelt",
    "spilt",
    "learnt",
    "told",
    "sold",
    "held",
    "found",
    "heard",
    "understood",
    "stood",
}

# words ending orthographically in -ed that are NOT a regular past
MONO_ED_EXCEPTIONS = {
    "need",
    "hundred",
    "sacred",
    "wicked",
    "naked",
    "crooked",
    "ragged",
    "jagged",
    "rugged",
    "wretched",
    "kindred",
    "hatred",
    "aged",
    "blessed",
    "beloved",
}


def normalize_word(word):
    """
    lowercase a word and strip surrounding punctuation, keeping internal apostrophes so contractions stay intact
    """
    if not isinstance(word, str):
        return ""
    w = word.lower().strip()
    return re.sub(r"^[^a-z']+|[^a-z']+$", "", w)


def is_neg_contraction(w):
    """
    true if the word is a negative contraction
    """
    if w.endswith("n't"):
        return True
    base = w.replace("’", "").replace("'", "")
    return any(base == stem or base == stem + "t" for stem in NEG_STEMS)


def classify_final_stop(word):
    """
    label the morphological status of a word-final coronal stop as one of mono, semiweak, past, neg_contraction, or function_word
    """
    w = normalize_word(word)
    if not w:
        return np.nan
    if is_neg_contraction(w):
        return "neg_contraction"
    if w in FUNCTION_WORDS:
        return "function_word"
    if w in SEMIWEAK:
        return "semiweak"
    if w in MONO_ED_EXCEPTIONS:
        return "mono"

    if w.endswith("ed") and len(w) > 3:
        return "past"
    return "mono"


def final_stop_morph_label(row):
    """
    morphological class of the final coronal stop
    """
    if normalize_feature(row.get("aae_feature", "")) not in TD_FEATURES:
        return NOT_APPLICABLE
    label = classify_final_stop(row.get("word", ""))
    return NOT_APPLICABLE if pd.isna(label) else label


def add_phonetic_and_morph_features(df: pd.DataFrame, zipfs_path: str) -> pd.DataFrame:
    """
    apply phoneme context, final-stop morphology, and frequency mapping to dataframe
    """
    df_enriched = df.copy()

    print("labeling words by phoneme context")
    context_df = df_enriched.apply(extract_phoneme_context, axis=1)
    df_enriched["previous_phoneme_manner"] = context_df["prev_manner"]
    df_enriched["next_phoneme_manner"] = context_df["next_manner"]
    df_enriched["previous_phoneme_place"] = context_df["prev_place"]
    df_enriched["next_phoneme_place"] = context_df["next_place"]
    df_enriched["previous_segment"] = context_df["prev_segment"]
    df_enriched["next_segment"] = context_df["next_segment"]

    print("labeling final-stop morphology in the t/d envelope")
    df_enriched["final_stop_morph"] = df_enriched.apply(final_stop_morph_label, axis=1)

    print("mapping lexical frequency")
    zipfs_df = pd.read_csv(zipfs_path)
    zipfs_dict = dict(
        zip(zipfs_df["Word"].astype(str).str.lower(), zipfs_df["Zipf-value"])
    )
    df_enriched["zipfs_frequency"] = (
        df_enriched["word"].astype(str).str.lower().map(zipfs_dict)
    )

    return df_enriched
