import pandas as pd


def generate_go_word_list(file):
    """
    function to create go word list from entire corpus to use for MAXQDA keyword-in-context search, which will be used to get corpus_context.csv
    """
    # import raw csv files
    df_codes_raw = pd.read_csv(file)

    # initialize go word list
    go_words = df_codes_raw["Segment"].unique()

    # create go words file
    with open("../../data/raw/corpus_go_words.txt", "w") as fp:
        for item in go_words:
            # write each item on a new line
            fp.write("%s\n" % item)
        print("Done")


if __name__ == "__main__":
    generate_go_word_list(file="../../data/raw/corpus_data.csv")
