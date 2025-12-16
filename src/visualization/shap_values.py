# %% imports
import shap
import pandas as pd
import numpy as np
import matplotlib
import seaborn as sns
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# %% load test data
X_test = pd.read_csv("../../data/processed/X_test.csv")
y_test = pd.read_csv("../../data/processed/y_test.csv")

# %% convert numerical features to float
num_features = ["zipfs_frequency"]
X_test[num_features] = X_test[num_features].astype(float)

# %% convert categorical features to category
cat_features = [c for c in X_test.columns if c not in num_features]
X_test[cat_features] = X_test[cat_features].astype("category")

# %% set ordered categories for aae_feature
aae_feature_order = ["ai monophthongization", "post-vocalic r", "post-consonantal t", "post-consonantal d", "ing ultimas", "third person singular", "auxiliary verb", "zero copula"]
if "aae_feature" in X_test.columns:
    aae_dtype = pd.CategoricalDtype(categories=aae_feature_order, ordered=True)
    X_test["aae_feature"] = X_test["aae_feature"].astype(aae_dtype)

# %% set ordered categories for next_phoneme_place
next_phoneme_place_order = ["vowel", "labiovelar", "bilabial", "labiodental", "dental", "alveolar", "postalveolar", "palatal", "velar", "glottal", "NONE"]
if "next_phoneme_place" in X_test.columns:
    next_phoneme_place_dtype = pd.CategoricalDtype(categories=next_phoneme_place_order, ordered=True)
    X_test["next_phoneme_place"] = X_test["next_phoneme_place"].astype(next_phoneme_place_dtype)

# %% set ordered categories for next_phoneme_manner
next_phoneme_manner_order = ["vowel", "glide", "liquid", "nasal", "obstruent", "NONE"]
if "next_phoneme_manner" in X_test.columns:
    next_phoneme_manner_dtype = pd.CategoricalDtype(categories=next_phoneme_manner_order, ordered=True)
    X_test["next_phoneme_manner"] = X_test["next_phoneme_manner"].astype(next_phoneme_manner_dtype)


# %% load trained model
model = CatBoostClassifier()
model.load_model("../../models/model.cbm")

# %% calculate and plot shap values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# %% keep a numeric copy of data for shap plotting (categorical codes) and a display copy for human-readable labels
X_test_numeric = X_test.copy()
for col in cat_features:
    X_test_numeric[col] = X_test_numeric[col].cat.codes.astype(float)
X_test_display = X_test.copy()

# %% use numeric data for plotting
shap_values.data = X_test_numeric.values
shap_values.display_data = X_test_display.values
shap_values.feature_names = list(X_test_numeric.columns)

# %% create subsets for each AAE feature
mask_ai = (X_test["aae_feature"] == "ai monophthongization").to_numpy()
mask_post_vocalic_r = (X_test["aae_feature"] == "post-vocalic r").to_numpy()
mask_post_consonantal_t = (X_test["aae_feature"] == "post-consonantal t").to_numpy()
mask_post_consonantal_d = (X_test["aae_feature"] == "post-consonantal d").to_numpy()
mask_ing_ultimas = (X_test["aae_feature"] == "ing ultimas").to_numpy()
mask_third_person_singular = (X_test["aae_feature"] == "third person singular").to_numpy()
mask_auxiliary_verb = (X_test["aae_feature"] == "auxiliary verb").to_numpy()
mask_zero_copula = (X_test["aae_feature"] == "zero copula").to_numpy()

shap_ai = shap_values[mask_ai]
shap_post_vocalic_r = shap_values[mask_post_vocalic_r]
shap_post_consonantal_t = shap_values[mask_post_consonantal_t]
shap_post_consonantal_d = shap_values[mask_post_consonantal_d]
shap_ing_ultimas = shap_values[mask_ing_ultimas]
shap_third_person_singular = shap_values[mask_third_person_singular]
shap_auxiliary_verb = shap_values[mask_auxiliary_verb]
shap_zero_copula = shap_values[mask_zero_copula]

# %% shap bar plot with custom colors
shap.plots.bar(shap_values, show=False, max_display=len(X_test.columns))

# define default shap colors and custom colors
default_pos_color = "#ff0051"
default_neg_color = "#008bfb"
positive_color = "#1F77B4"
negative_color = "#D0E2F2"

# recolor rectangles and texts
for fc in plt.gcf().get_children():
    # ignore last rectangle (legend background)
    for fcc in fc.get_children()[:-1]:
        if isinstance(fcc, matplotlib.patches.Rectangle):
            face_hex = matplotlib.colors.to_hex(fcc.get_facecolor())
            if face_hex == default_pos_color:
                fcc.set_facecolor(positive_color)
            elif face_hex == default_neg_color:
                fcc.set_facecolor(negative_color)
        elif isinstance(fcc, plt.Text):
            text_hex = matplotlib.colors.to_hex(fcc.get_color())
            if text_hex == default_pos_color:
                fcc.set_color(positive_color)
            elif text_hex == default_neg_color:
                fcc.set_color(negative_color)
plt.xlabel("mean(|SHAP value|)")
plt.savefig("../../figures/shap/shap_mean_absolute_bar_plot.png", dpi=600, bbox_inches="tight")
plt.show()

# %% beeswarm with absolute shap values with custom colors
shap.plots.beeswarm(shap_values.abs, show=False, color=positive_color, max_display=len(X_test.columns))
plt.xlabel("|SHAP value|")
plt.savefig("../../figures/shap/shap_absolute_beeswarm_plot.png", dpi=600, bbox_inches="tight")
plt.show()

# %% waterfall for single datapoint plot with custom colors
datapoint_index = 200
shap.plots.waterfall(shap_values[datapoint_index], show=False, max_display=len(X_test.columns))

for fc in plt.gcf().get_children():
    for fcc in fc.get_children():
        if (isinstance(fcc, matplotlib.patches.FancyArrow)):
            if (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_pos_color):
                fcc.set_facecolor(positive_color)
                fcc.set_edgecolor(positive_color)
            elif (matplotlib.colors.to_hex(fcc.get_facecolor()) == default_neg_color):
                fcc.set_color(negative_color)   
        elif (isinstance(fcc, plt.Text)):
            if (matplotlib.colors.to_hex(fcc.get_color()) == default_pos_color):
                fcc.set_color(positive_color)
            elif (matplotlib.colors.to_hex(fcc.get_color()) == default_neg_color):
                fcc.set_color(negative_color)
plt.savefig("../../figures/shap/shap_waterfall_plot_for_datapoint_200.png", dpi=600, bbox_inches="tight")
plt.show()

# %% scatterplot for all features by zipfs_frequency
shap.plots.scatter(
shap_values[:, "aae_feature"],
color=shap_values[:, "zipfs_frequency"],
show=False,
cmap=sns.color_palette("flare", as_cmap=True)
)
plt.xticks(rotation=45, ha="right")
plt.title("SHAP values by aae_feature and zipfs_frequency")
plt.ylabel("SHAP value")
plt.savefig("../../figures/shap/shap_values_by_aae_feature_and_zipfs_frequency.png", dpi=600, bbox_inches="tight")
plt.show()

# %% scatterplot for ai monophthongization by zipfs_frequency
shap.plots.scatter(
shap_ai[:, "zipfs_frequency"],
color=shap_ai[:, "zipfs_frequency"],
cmap=sns.color_palette("flare", as_cmap=True),
show=False,
)
plt.xticks(rotation=45, ha="right")
plt.title("SHAP values for PRICE monophthongization by zipfs_frequency")
plt.ylabel("SHAP value")
plt.savefig("../../figures/shap/shap_values_for_ai_monophthongization_by_zipfs_frequency.png", dpi=600, bbox_inches="tight")
plt.show()

# %% scatterplot for t deletion by zipfs_frequency
shap.plots.scatter(
shap_post_consonantal_t[:, "zipfs_frequency"],
show=False
)
plt.ylim(-1, 1)
plt.xticks(rotation=45, ha="right")
plt.title("SHAP values for t deletion by zipfs_frequency")
plt.ylabel("SHAP value")
plt.show()

# %% scatterplot for d deletion by zipfs_frequency
shap.plots.scatter(
shap_post_consonantal_d[:, "zipfs_frequency"],
show=False,
)
plt.xticks(rotation=45, ha="right")
plt.title("SHAP values for d deletion by zipfs_frequency")
plt.ylabel("SHAP value")
plt.show()

# %% scatterplot for all features by next_phoneme_place
shap.plots.scatter(
shap_values[:, "aae_feature"],
color=shap_values[:, "next_phoneme_place"],
show=False,
cmap=sns.color_palette("Spectral", as_cmap=True)
)
plt.xticks(rotation=45, ha="right")
plt.title("SHAP values by aae_feature and next_phoneme_place")
plt.ylabel("SHAP value")
plt.savefig("../../figures/shap/shap_values_by_aae_feature_and_next_phoneme_place.png", dpi=600, bbox_inches="tight")
plt.show()

# %% scatterplot for all features by next_phoneme_manner
shap.plots.scatter(
shap_values[:, "aae_feature"],
color=shap_values[:, "next_phoneme_manner"],
show=False,
cmap=sns.color_palette("Spectral", as_cmap=True)
)
plt.xticks(rotation=45, ha="right")
plt.title("SHAP values by aae_feature and next_phoneme_manner")
plt.ylabel("SHAP value")
plt.savefig("../../figures/shap/shap_values_by_aae_feature_and_next_phoneme_manner.png", dpi=600, bbox_inches="tight")
plt.show()

# %% scatterplot for all features by word_morphemes
# select only mono and bimorphemic words
mask_mono_bi = (X_test["word_morphemes"].isin(["mono", "bi"])).to_numpy()
shap_values_mono_bi = shap_values[mask_mono_bi]

shap.plots.scatter(
shap_values_mono_bi[:, "aae_feature"],
color=shap_values_mono_bi[:, "word_morphemes"],
show=False,
cmap=sns.color_palette("Spectral", as_cmap=True)
)
plt.xticks(rotation=45, ha="right")
plt.title("SHAP values by aae_feature and word_morphemes")
plt.ylabel("SHAP value")
plt.savefig("../../figures/shap/shap_values_by_aae_feature_and_word_morphemes.png", dpi=600, bbox_inches="tight")
plt.show()

# %%
