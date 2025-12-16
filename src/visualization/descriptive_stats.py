# %% imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# set seaborn style
sns.set_style("white")
sns.set_context("talk")

df = pd.read_csv("../../data/processed/corpus_data_processed.csv")

# %% subset of features
phonetic_features = [
    "ing ultimas",
    "ai monophthongization",
    "post-vocalic r",
    "post-consonantal d",
    "post-consonantal t",
]

grammatical_features = [
    "auxiliary verb",
    "third person singular",
    "zero copula",
]

phonetic_df = df[df["aae_feature"].isin(phonetic_features)]
grammatical_df = df[df["aae_feature"].isin(grammatical_features)]

# %% shorter feature names 
short_feature_names = {
    "ing ultimas": "/ɪn/",
    "ai monophthongization": "/ai/",
    "post-vocalic r": "/r/",
    "post-consonantal d": "/d/",
    "post-consonantal t": "/t/",
    "auxiliary verb": "ain't",
    "third person singular": "3rd person",
    "zero copula": "copula",
}

df["aae_feature"] = df["aae_feature"].map(short_feature_names)
phonetic_df["aae_feature"] = phonetic_df["aae_feature"].map(short_feature_names)
grammatical_df["aae_feature"] = grammatical_df["aae_feature"].map(short_feature_names)

# %% mean outcome by artist and group
df["group"] = df["time"] + df["social_group"]

# artist means
blues_artist_mean = (
    df.groupby(["artist", "group"])["aae_realization"]
    .mean()
    .reset_index()
    .sort_values(by="aae_realization", ascending=False)
)  # calculate group means

custom_colors = sns.color_palette("muted", 9)  # define custom colors for each group

groups = [
    "1960sAA",
    "1960snonAA_US",
    "1960snonAA_nonUS",
    "1980sAA",
    "1980snonAA_US",
    "1980snonAA_nonUS",
    "2010sAA",
    "2010snonAA_US",
    "2010snonAA_nonUS",
]

g = sns.catplot(
    data=blues_artist_mean,
    x="aae_realization",
    y="artist",
    hue="group",
    hue_order=groups,
    palette=custom_colors,
    height=13.5,
    aspect=1.75,
    s=75,
    legend=False,
)  # create the catplot

group_means = blues_artist_mean.groupby("group")["aae_realization"].mean()
for group, color in zip(group_means.index, custom_colors):
    mean_value = group_means[group]
    plt.axvline(
        mean_value, color=color, linestyle="--", linewidth=3
    )  # calculate group means and add as horizontal lines

handles = [
    mpatches.Patch(color=color, label=group)
    for color, group in zip(custom_colors, groups)
]
plt.legend(
    handles=handles,
    title="group",
    bbox_to_anchor=(0.9, 0.75),
    fontsize=20,
    title_fontsize=25,
    frameon=False,
)

g.set_axis_labels("mean AAE realization", "artist", fontsize=25)
g.set_xticklabels(fontsize=22.5)
g.set_yticklabels(fontsize=20)

# set x axis ticks
xticks = [round(x, 2) for x in list(np.arange(0.5, 0.96, 0.05))]
plt.xticks(xticks, [f"{tick}" for tick in xticks], fontsize=22.5)

plt.tight_layout()
plt.xlim(0.5, 1)
plt.savefig("../../figures/descriptive/mean_outcome_by_artist_and_group.png", dpi=600)
plt.show()

# %% mean outcome by feature and group
group_order = [
        "1960sAA",
        "1960snonAA_US",
        "1960snonAA_nonUS",
        "1980sAA",
        "1980snonAA_US",
        "1980snonAA_nonUS",
        "2010sAA",
        "2010snonAA_US",
        "2010snonAA_nonUS",
    ]

feature_order = [
    "/ɪn/",
    "/ai/",
    "/r/",
    "/d/",
    "/t/",
    "ain't",
    "3rd person",
    "copula",
]

g = sns.catplot(
    data=df,
    y="aae_feature",
    x="aae_realization",
    col="group",
    col_wrap=3,
    col_order=group_order,
    order=feature_order,
    kind="point",
    errorbar="ci",
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("AAE feature")
plt.savefig("../../figures/descriptive/mean_outcome_by_feature_and_group.png", dpi=600)
plt.show()

# %% mean outcome by feature, group and song type
g = sns.catplot(
    data=df,
    y="aae_feature",
    x="aae_realization",
    col="group",
    col_wrap=3,
    hue="type",
    col_order=group_order,
    order=feature_order,
    kind="point",
    errorbar="ci",
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("AAE feature")
plt.savefig("../../figures/descriptive/mean_outcome_by_feature_group_and_song_type.png", dpi=600)
plt.show()

# %% mean outcome by feature, group and performance type
g = sns.catplot(
    data=df,
    y="aae_feature",
    x="aae_realization",
    col="group",
    col_wrap=3,
    hue="performance",
    col_order=group_order,
    order=feature_order,
    kind="point",
    errorbar="ci",
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("AAE feature")
plt.savefig("../../figures/descriptive/mean_outcome_by_feature_group_and_performance_type.png", dpi=600)
plt.show()

# %% mean outcome by phonetic feature and social group
phonetic_feature_order = sorted(phonetic_df["aae_feature"].unique(), reverse=False)
plt.figure(figsize=(10, 5))
sns.pointplot(
    data=phonetic_df, 
    x="aae_feature", 
    order=phonetic_feature_order,
    y="aae_realization", 
    hue="social_group", 
    dodge=False, 
    errorbar="ci"
)
sns.despine()
plt.xlabel("")
plt.ylabel("")
plt.legend(loc="lower right", frameon=False)
plt.ylim(0, 1)
plt.savefig("../../figures/descriptive/mean_outcome_by_phonetic_feature_and_social_group.png", dpi=600)
plt.show()

# %% mean outcome by phonetic feature, context and social group
phonetic_feature_order = sorted(phonetic_df["aae_feature"].unique(), reverse=False)
col_order = sorted(phonetic_df["type"].unique(), reverse=True)
row_order = sorted(phonetic_df["performance"].unique(), reverse=True)

g = sns.FacetGrid(
    phonetic_df,
    row="performance",
    col="type",
    hue="social_group",
    height=4,
    aspect=1.75,
    margin_titles=True,
    ylim=(0, 1),
    row_order=row_order,
    col_order=col_order,
)
g.map(
    sns.pointplot,
    "aae_feature",
    "aae_realization",
    order=phonetic_feature_order,
    errorbar="ci",
    dodge=False
)
g.add_legend(title="")
g.set_axis_labels("", "")
for ax in g.axes.flatten():
    ax.set_ylim(0, 1)
plt.savefig("../../figures/descriptive/mean_outcome_by_phonetic_feature_context_and_social_group.png", dpi=600)
plt.show()

# %% mean outcome by grammatical feature, context and social group
grammatical_feature_order = sorted(grammatical_df["aae_feature"].unique(), reverse=False)
col_order = sorted(grammatical_df["type"].unique(), reverse=True)
row_order = sorted(grammatical_df["performance"].unique(), reverse=True)

g = sns.FacetGrid(
    grammatical_df,
    row="performance",
    col="type",
    hue="social_group",
    height=4,
    aspect=1.75,
    margin_titles=True,
    ylim=(0, 1.1),
    row_order=row_order,
    col_order=col_order,
)
g.map(
    sns.pointplot,
    "aae_feature",
    "aae_realization",
    order=grammatical_feature_order,
    errorbar="ci",
    dodge=False
)
g.add_legend(title="")
g.set_axis_labels("", "")
for ax in g.axes.flatten():
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
plt.savefig("../../figures/descriptive/mean_outcome_by_grammatical_feature_context_and_social_group.png", dpi=600)
plt.show()

# %% mean outcome by time, context and social group
g = sns.FacetGrid(
    df,
    row="performance",
    col="type",
    hue="social_group",
    height=4,
    aspect=1.75,
    margin_titles=True,
    ylim=(0, 1),
    row_order=row_order,
    col_order=col_order,
)
g.map(
    sns.pointplot,
    "time",
    "aae_realization",
    errorbar="ci",
    dodge=False
)
g.add_legend(title="")
g.set_axis_labels("", "")
for ax in g.axes.flatten():
    ax.set_ylim(0, 1)
plt.savefig("../../figures/descriptive/mean_outcome_by_time_context_and_social_group.png", dpi=600)
plt.show()


# %% mean outcome by artist, group and song type
g = sns.catplot(
    data=df,
    y="artist",
    x="aae_realization",
    col="group",
    col_wrap=3,
    hue="type",
    col_order=[
        "1960sAA",
        "1960snonAA_US",
        "1960snonAA_nonUS",
        "1980sAA",
        "1980snonAA_US",
        "1980snonAA_nonUS",
        "2010sAA",
        "2010snonAA_US",
        "2010snonAA_nonUS",
    ],
    kind="point",
    errorbar="ci",
    sharey=False,
    aspect=1.25,
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("artist")
g.set(
    xlim=(0, 1),
    xticks=[0, 0.25, 0.5, 0.75, 1],
    xticklabels=["0", "0,25", "0.5", "0.75", "1"],
)
plt.savefig("../../figures/descriptive/mean_outcome_by_artist_group_and_song_type.png", dpi=600)
plt.show()


# %% mean outcome by artist, group and performance type
g = sns.catplot(
    data=df,
    y="artist",
    x="aae_realization",
    col="group",
    col_wrap=3,
    hue="performance",
    col_order=[
        "1960sAA",
        "1960snonAA_US",
        "1960snonAA_nonUS",
        "1980sAA",
        "1980snonAA_US",
        "1980snonAA_nonUS",
        "2010sAA",
        "2010snonAA_US",
        "2010snonAA_nonUS",
    ],
    kind="point",
    errorbar="ci",
    sharey=False,
    aspect=1.25,
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("artist")
g.set(
    xlim=(0, 1),
    xticks=[0, 0.25, 0.5, 0.75, 1],
    xticklabels=["0", "0,25", "0.5", "0.75", "1"],
)
plt.savefig(
    "../../figures/descriptive/mean_outcome_by_artist_group_and_performance_type.png", dpi=600
)
plt.show()

# %% r deletion by next phoneme and social group
df_r_deletion = phonetic_df[phonetic_df["aae_feature"] == "/r/"].copy()

def classify_next_segment(x):
    if pd.isnull(x):
        return np.nan
    return "vowel" if x == "vowel" else "consonant"

df_r_deletion["next_segment"] = df_r_deletion["next_phoneme_manner"].apply(classify_next_segment)

plt.figure(figsize=(7, 7))
sns.pointplot(
    data=df_r_deletion,
    x="next_segment",
    y="aae_realization",
    hue="social_group",
    errorbar="ci",
    dodge=False,
)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("/r/ deletion by next segment and social group")
plt.xlabel("next segment")
plt.ylabel("/r/ deletion")
plt.legend(loc="lower right", frameon=False)
plt.tight_layout()
plt.savefig("../../figures/descriptive/r_deletion_by_next_segment_and_social_group.png", dpi=600)
plt.show()

# %% t deletion by word morphemes and social group
df_t_deletion = phonetic_df[phonetic_df["aae_feature"] == "/t/"].copy()
# remove tri+ words
df_t_deletion = df_t_deletion[df_t_deletion["word_morphemes"] != "tri+"]

plt.figure(figsize=(7, 7))
sns.pointplot(
    data=df_t_deletion,
    x="word_morphemes",
    y="aae_realization",
    hue="social_group",
    errorbar="ci",
    dodge=False,
    order=["mono", "bi"],
)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("/t/ deletion by word morphemes and social group")
plt.xlabel("word morphemes")
plt.ylabel("/t/ deletion")
plt.legend(loc="best", frameon=False)
plt.tight_layout()
plt.savefig("../../figures/descriptive/t_deletion_by_word_morphemes_and_social_group.png", dpi=600)
plt.show()

# %% d deletion by word morphemes and social group
df_d_deletion = phonetic_df[phonetic_df["aae_feature"] == "/d/"].copy()
# remove tri+ words
df_d_deletion = df_d_deletion[df_d_deletion["word_morphemes"] != "tri+"]

plt.figure(figsize=(7, 7))
sns.pointplot(
    data=df_d_deletion,
    x="word_morphemes",
    y="aae_realization",
    hue="social_group",
    errorbar="ci",
    dodge=False,
    order=["mono", "bi"],
)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("/d/ deletion by word morphemes and social group")
plt.xlabel("word morphemes")
plt.ylabel("/d/ deletion")
plt.legend(loc="best", frameon=False)
plt.tight_layout()
plt.savefig("../../figures/descriptive/d_deletion_by_word_morphemes_and_social_group.png", dpi=600)
plt.show()

# %% plot most frequent words
top_words_df = df["word"].value_counts().head(50).reset_index()
top_words_df.columns = ["word", "absolute_frequency"]

plt.figure(figsize=(25, 10))
sns.barplot(
    data=top_words_df,
    x="word",
    y="absolute_frequency",
)
plt.xticks(rotation=45, ha="right")
plt.title("50 most frequent words")
plt.savefig("../../figures/descriptive/top_50_most_frequent_words.png", dpi=600)
plt.show()