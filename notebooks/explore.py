# %% imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% set seaborn style
sns.set_style("white")
sns.set_context("notebook")

# %% import data
df = pd.read_csv("../data/interim/corpus_data_pre_processed.csv")
df.head()

# %% check outcome distribution for each AAE feature
aae_feature_table = (
    df.groupby("aae_feature")["aae_realization"]
    .value_counts()
    .unstack(fill_value=0)
)

print("absolute counts:")
print(aae_feature_table)

aae_feature_rel = aae_feature_table.div(aae_feature_table.sum(axis=1), axis=0)
print("\nrelative frequencies (row-wise):")
print(aae_feature_rel)

# %% check overall outcome distribution
print(len(df))
print(df["aae_realization"].value_counts())
print(df["aae_realization"].value_counts(normalize=True))

# %% check unique words
df["word"].nunique()

# %% check top 25 words
df["word"].value_counts().head(25)

# %% univariate plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.pointplot(data=df, x="time", y="aae_realization", ax=axes[0, 0])
sns.despine(ax=axes[0, 0])
axes[0, 0].set_ylim(0.5, 1)
axes[0, 0].set_title("aae_realization by time")
plt.setp(axes[0, 0].get_xticklabels(), rotation=45, ha="right")

sns.pointplot(data=df, x="performance", y="aae_realization", ax=axes[0, 1])
sns.despine(ax=axes[0, 1])
axes[0, 1].set_ylim(0.5, 1)
axes[0, 1].set_title("aae_realization by performance")
plt.setp(axes[0, 1].get_xticklabels(), rotation=45, ha="right")

sns.pointplot(data=df, x="type", y="aae_realization", ax=axes[1, 0])
sns.despine(ax=axes[1, 0])
axes[1, 0].set_ylim(0.5, 1)
axes[1, 0].set_title("aae_realization by type")
plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha="right")

sns.pointplot(data=df, x="social_group", y="aae_realization", ax=axes[1, 1])
sns.despine(ax=axes[1, 1])
axes[1, 1].set_ylim(0.5, 1)
axes[1, 1].set_title("aae_realization by social_group")
plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.show()

# %% pairgrid plot for time, performance, type by social_group
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
x_vars = ["time", "performance", "type"]

for ax, x in zip(axes, x_vars):
    sns.pointplot(data=df, x=x, y="aae_realization", hue="social_group", ax=ax)
    sns.despine(ax=ax)
    ax.set_title(f"aae_realization by {x}")
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# keep a single legend (remove duplicates from the first two axes)
for ax in axes[:-1]:
    legend = ax.get_legend()
    if legend:
        legend.remove()

# move the remaining legend to lower right
handles, labels = axes[-1].get_legend_handles_labels()
axes[-1].legend(
    handles=handles,
    labels=labels,
    loc="lower right",
    title="social_group",
    frameon=False,
)

plt.tight_layout()
plt.show()

# %% point plot: aae_feature
plt.figure(figsize=(10, 5))
sns.pointplot(
    data=df,
    x="aae_feature",
    y="aae_realization",
    hue="social_group",
    dodge=False,
    errorbar="ci",
    markers=["o", "s", "d"],
    linestyles=["-", "--", ":"],
)
sns.despine()
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("aae realization by feature and social group")
plt.legend(title="social group", loc="lower left", frameon=False)
plt.tight_layout()
plt.show()

# %% point plot: three-way interaction (social_group x performance x type)
g = sns.FacetGrid(
    df,
    col="type",
    row="performance",
    hue="social_group",
    height=4,
    aspect=1,
    margin_titles=True,
)
g.map(sns.pointplot, "time", "aae_realization", errorbar="ci", dodge=True)
g.set(ylim=(0, 1))
plt.legend(title="social group", loc="lower right", frameon=False)
plt.suptitle("three-way interaction: social group x performance x song type over time")
plt.tight_layout()
plt.show()

# %% point plot: performance by time and social group (faceted)
g = sns.FacetGrid(
    df, col="time", hue="social_group", height=5, aspect=0.8, margin_titles=True
)
g.map(sns.pointplot, "performance", "aae_realization", errorbar="ci", dodge=True)
g.set(ylim=(0, 1))
plt.legend(title="social group", loc="lower right", frameon=False)
plt.suptitle("aae realization by performance type: social group x time")
plt.tight_layout()
plt.show()

# %% top words analysis
top_words = df["word"].value_counts().head(15).index
df_top_words = df[df["word"].isin(top_words)]

plt.figure(figsize=(12, 6))
sns.pointplot(
    data=df_top_words,
    x="word",
    y="aae_realization",
    hue="social_group",
    errorbar="ci",
    dodge=True,
)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("aae realization for top 15 most frequent words")
plt.legend(title="social group", loc="lower left", frameon=False)
plt.tight_layout()
plt.show()

# %% point plot: type by time for each social group (faceted)
g = sns.FacetGrid(
    df, col="social_group", hue="time", height=5, aspect=0.7, margin_titles=True
)
g.map(sns.pointplot, "type", "aae_realization", errorbar="ci", dodge=True)
g.set(ylim=(0, 1))
plt.legend(title="social group", loc="lower right", frameon=False)
plt.suptitle("aae realization by song type: time x social group")
plt.tight_layout()
plt.show()

# %% gender comparison
df_gender = df

female_artists = [
    "janisjoplin",
    "kokotaylor",
    "bonnirraitt",
    "shemekiacopeland",
    "allyvenable",
    "samanthafish",
    "daniwilde",
    "joanneshawwtaylor",
]

df_gender["gender"] = df_gender["artist"].isin(female_artists).map({True: "female", False: "male"})
df_gender.head()

# %% point plot: aae_feature by gender
plt.figure(figsize=(10, 5))
sns.pointplot(
    data=df_gender,
    x="aae_feature",
    y="aae_realization",
    hue="gender",
    errorbar="ci",
    dodge=False,
)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("aae realization by feature and gender")

# %% quick and dirty vowel and consonant classification
vowels = set("aeiou")


def classify_final_char(token):
    # return vowel/consonant/none for last char
    if not isinstance(token, str) or not token:
        return np.nan
    last_char = token.strip().lower()[-1]
    if last_char.isalpha():
        return "vowel" if last_char in vowels else "consonant"
    return np.nan


def classify_initial_char(token):
    # return vowel/consonant/none for first char
    if not isinstance(token, str) or not token:
        return np.nan
    first_char = token.strip().lower()[0]
    if first_char.isalpha():
        return "vowel" if first_char in vowels else "consonant"
    return np.nan


df["postvocalic"] = df["previous_word"].apply(classify_final_char)
df["prevocalic"] = df["next_word"].apply(classify_initial_char)

# %% plot r-deletion by prevocalic and social group
df_r_deletion = df[df["aae_feature"] == "post-vocalic r"]
plt.figure(figsize=(5, 5))
sns.pointplot(
    data=df_r_deletion,
    x="prevocalic",
    y="aae_realization",
    hue="social_group",
    errorbar="ci",
    dodge=False,
)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1)
plt.title("aae realization by r-deletion and social group")
plt.tight_layout()
plt.show()
# %%
