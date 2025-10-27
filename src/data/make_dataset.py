import pandas as pd
import re
import unicodedata


def replace_apostrophes(text):
    """
    function to fix curly apostrophes
    """
    if isinstance(text, str):
        text = text.replace("’", "'")
        text = text.replace("‘", "'")
    return text


def normalize_key(text):
    """
    normalize whitespace and unicode for join keys
    """
    if isinstance(text, str):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u00a0", " ")
        text = re.sub(r"\s+", " ", text).strip()
    return text


def make_data(data_raw, context_raw, metadata_raw):
    """
    function to make processed data set after keyword-in-context search has been performed in MAXQDA
    """

    # drop rows from data_raw
    data_raw.drop(
        columns=[
            "End",
            "Modified by",
            "Created by",
            "Color",
            "Weight score",
            "Modified",
            "Code alias",
        ],
        inplace=True,
    )

    # rename context columns
    context_raw.rename(
        columns={
            "Context": "Previous word",
            "Context.1": "Next word",
            "Keyword": "Segment",
        },
        inplace=True,
    )

    # replace apostrophes
    data_raw = data_raw.map(replace_apostrophes)
    context_raw = context_raw.map(replace_apostrophes)

    # replace NAs in context columns with empty strings
    context_raw["Previous word"] = context_raw["Previous word"].fillna("")
    context_raw["Next word"] = context_raw["Next word"].fillna("")

    # set previous word ending in 2 spaces to empty string
    mask_pre_context = context_raw["Previous word"].str.endswith("  ")
    context_raw.loc[mask_pre_context, "Previous word"] = ""

    # set next word starting in 2 spaces to empty string
    mask_post_context = context_raw["Next word"].str.startswith("  ")
    context_raw.loc[mask_post_context, "Next word"] = ""

    # normalize join keys (segment/keyword) to avoid whitespace/newline mismatches
    data_raw["Segment"] = data_raw["Segment"].map(normalize_key)
    context_raw["Segment"] = context_raw["Segment"].map(normalize_key)

    # merge raw csv files
    df_merged = data_raw.merge(
        context_raw,
        how="left",
        on=["Document name", "Beginning", "Segment", "Document group"],
    )

    # if a word occurs more than once in the same line of the lyrics, duplicate rows will be created
    # using the unique Created column to drop these rows
    df_merged.drop_duplicates(subset="Created", keep="first", inplace=True)

    # rename segment column
    df_merged.rename(
        columns={
            "Segment": "Word",
        },
        inplace=True,
    )

    # fill nas in merged context columns
    df_merged["Previous word"] = df_merged["Previous word"].fillna("")
    df_merged["Next word"] = df_merged["Next word"].fillna("")

    # strip context columns
    df_merged["Previous word"] = df_merged["Previous word"].str.strip()
    df_merged["Next word"] = df_merged["Next word"].str.strip()

    # keep only last word in previous word column
    df_merged["Previous word"] = df_merged["Previous word"].str.split().str[-1]

    # keep only first word in next word column
    df_merged["Next word"] = df_merged["Next word"].str.split().str[0]

    # if previous word ends with double parenthesis (meta comments), make cell empty
    df_merged.loc[
        df_merged["Previous word"].str.endswith("))", na=False), "Previous word"
    ] = ""

    # if next word starts with double parenthesis (meta comments), make cell empty
    df_merged.loc[
        df_merged["Next word"].str.startswith("((", na=False), "Next word"
    ] = ""

    # now remove all single parentheses (uncertain next word) from both columns
    df_merged["Previous word"] = df_merged["Previous word"].str.replace(r"\(|\)", "")
    df_merged["Next word"] = df_merged["Next word"].str.replace(r"\(|\)", "")

    # unpack document name
    df_merged[["Artist", "Performance", "Song"]] = df_merged[
        "Document name"
    ].str.extract(r"(\w+)-(\w+)-([\w_]+)", expand=True)

    # unpack code
    df_merged[["Variable", "Value"]] = df_merged["Code"].str.split(" > ", expand=True)

    # rename performance context
    performance_contexts = {
        "so": "studio-original",
        "sc": "studio-cover",
        "lo": "live-original",
        "lc": "live-cover",
    }
    df_merged.replace({"Performance": performance_contexts}, inplace=True)

    # unpack performance context
    df_merged[["Performance", "Type"]] = df_merged["Performance"].str.split(
        "-", expand=True
    )

    # unpack time and social group
    df_merged[["Time", "Social group"]] = df_merged["Document group"].str.extract(
        r"(\d{4}s)_(\w+(?:_\w+)?)"
    )

    # make value column binary
    value_binary = {
        "a:": 1,
        "ai": 0,
        "t deletion": 1,
        "t realization": 0,
        "d deletion": 1,
        "d realization": 0,
        "r deletion": 1,
        "r realization": 0,
        "in": 1,
        "ing": 0,
        "ain't": 1,
        "isn't": 0,
        "s deletion": 1,
        "s realization": 0,
        "copula deletion": 1,
        "copula realization": 0,
    }
    df_merged.replace({"Value": value_binary}, inplace=True)

    # drop redundant columns
    df_merged.drop(
        columns=["Document name", "Code", "Beginning", "Document group"],
        inplace=True,
    )

    # prepare metadata df for merging
    metadata_raw = metadata_raw[["artist", "song", "year", "type", "performance"]]

    # merge to add year
    df_final = df_merged.merge(
        metadata_raw,
        how="left",
        left_on=["Artist", "Song", "Performance", "Type"],
        right_on=["artist", "song", "performance", "type"],
    )

    # drop duplicate rows again
    df_final.drop_duplicates(subset="Created", keep="first", inplace=True)

    # drop redundant columns
    df_final.drop(
        columns=["Created", "artist", "song", "type", "performance"], inplace=True
    )

    # convert columns names to snake case
    df_final.rename(columns=lambda x: x.lower().replace(" ", "_"), inplace=True)

    # convert variable and value column names
    df_final.rename(
        columns={"variable": "aae_feature", "value": "aae_realization"}, inplace=True
    )

    # export interim dataframe
    df_final.to_csv(
        "../../data/interim/corpus_data_pre_processed.csv",
        index=False,
        encoding="UTF-8",
    )


if __name__ == "__main__":
    data_raw = pd.read_csv("../../data/raw/corpus_data.csv")
    context_raw = pd.read_csv("../../data/raw/corpus_context.csv")
    metadata_raw = pd.read_csv("../../data/raw/corpus_metadata.csv")
    make_data(
        data_raw=data_raw,
        context_raw=context_raw,
        metadata_raw=metadata_raw,
    )
