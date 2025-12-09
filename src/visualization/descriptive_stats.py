# %% imports
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# set seaborn style
sns.set_style("white")
sns.set_context("talk")

df = pd.read_csv("../../data/interim/corpus_data_pre_processed.csv")

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

# %% artist and group mean plot
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
plt.savefig("../../figures/mean_aae_realizations.png", dpi=600)
plt.show()

# %% point plot by group
g = sns.catplot(
    data=df,
    y="aae_feature",
    x="aae_realization",
    col="group",
    col_wrap=3,
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
    order=[
        "ing ultimas",
        "ai monophthongization",
        "post-vocalic r",
        "post-consonantal d",
        "post-consonantal t",
        "auxiliary verb",
        "third person singular",
        "zero copula",
    ],
    kind="point",
    errorbar="ci",
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("AAE feature")
plt.savefig("../../figures/point_plots_by_group.png", dpi=600)
plt.show()

# %% point plot by group and song type
g = sns.catplot(
    data=df,
    y="aae_feature",
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
    order=[
        "ing ultimas",
        "ai monophthongization",
        "post-vocalic r",
        "post-consonantal d",
        "post-consonantal t",
        "auxiliary verb",
        "third person singular",
        "zero copula",
    ],
    kind="point",
    errorbar="ci",
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("AAE feature")
plt.savefig("../../figures/point_plots_by_group_and_song_type.png", dpi=600)
plt.show()

# %% point plot by group and performance
g = sns.catplot(
    data=df,
    y="aae_feature",
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
    order=[
        "ing ultimas",
        "ai monophthongization",
        "post-vocalic r",
        "post-consonantal d",
        "post-consonantal t",
        "auxiliary verb",
        "third person singular",
        "zero copula",
    ],
    kind="point",
    errorbar="ci",
)
g.set_xlabels("mean AAE realization")
g.set_ylabels("AAE feature")
plt.savefig("../../figures/point_plots_by_group_and_performance_type.png", dpi=600)
plt.show()

# %% outcome by feature and social group
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
plt.savefig("../../figures/outcome_by_phonetic_feature_and_social_group.png", dpi=600)
plt.show()

# %% 3-way plots for phonetic features

# Get x (aae_feature) and y (performance) in reverse alphabetical order
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
plt.savefig("../../figures/outcome_by_phonetic_feature_context_and_social_group.png", dpi=600)
plt.show()

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
    "time",
    "aae_realization",
    errorbar="ci",
    dodge=False
)
g.add_legend(title="")
g.set_axis_labels("", "")
for ax in g.axes.flatten():
    ax.set_ylim(0, 1)
plt.savefig("../../figures/outcome_by_phonetic_time_context_and_social_group.png", dpi=600)
plt.show()

# %% 3-way plots for grammatical features

# Get x (aae_feature) and y (performance) in reverse alphabetical order
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
plt.savefig("../../figures/outcome_by_grammatical_feature_context_and_social_group.png", dpi=600)
plt.show()


# %% point plot by group, artist and type
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
plt.savefig("../../figures/point_plots_by_group_artist_and_song_type.png", dpi=600)
plt.show()


# point plot by group, artist and song type
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
    "../../figures/point_plots_by_group_artist_and_performance_type.png", dpi=600
)
plt.show()


# aae realization by group across time
g = sns.catplot(
    data=df,
    x="aae_feature",
    y="aae_realization",
    hue="social_group",
    kind="point",
    errorbar="ci",
    height=13.5,
    aspect=1.75,
)
g.set_axis_labels("AAE feature", "AAE realization")
g.set_xticklabels(fontsize=22.5)
g.set_yticklabels(fontsize=20)
plt.savefig("../../figures/aae_realization_by_group_across_time.png", dpi=600)
plt.tight_layout()
plt.show()

# %% mean outcome by social_group across contexts (type x performance)
g = sns.catplot(
    data=df,
    x="aae_feature",
    y="aae_realization",
    col="performance",
    row="type",
    kind="point",
    errorbar="ci",
    height=4,
    aspect=1.2,
)
g.set_axis_labels("AAE feature", "AAE realization")
g.set_titles(row_template="type: {row_name}", col_template="performance: {col_name}")
plt.tight_layout()
plt.show()
