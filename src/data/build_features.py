import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# set seed
seed = 42
np.random.seed(seed)

# import data
df = pd.read_csv("../../data/interim/corpus_data_pre_processed.csv")
df.head()

# check NAs in categorical columns
cols = df.columns
cat_cols = [col for col in df.columns if df[col].dtype == "object"]
df.isna().sum()

# fill NAs in categorical columns
for col in cat_cols:
    df[col] = df[col].fillna("")
df.isna().sum()

# create song identifier for grouping
# ensures entire songs stay together in splits
df["song_id"] = df["artist"] + "_" + df["song"] + "_" + df["performance"]
song_groups = df["song_id"]

# prepare features and target
y = df["aae_realization"]
# drop song_id (grouping variable) and song (prevents data leakage in song-based splits)
# keep artist to allow learning artist-level patterns across different songs
X = df.drop(columns=["aae_realization", "song_id", "song"])

print("=== dataset overview ===")
print(f"total tokens: {len(df)}")
print(f"unique songs: {df['song_id'].nunique()}")
print(f"class distribution: {y.value_counts(normalize=True).to_dict()}")
print()

# first split: separate test set (15% of songs)
gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
train_val_idx, test_idx = next(gss_test.split(X, y, groups=song_groups))

X_train_temp = X.iloc[train_val_idx]
y_train_temp = y.iloc[train_val_idx]
song_groups_train_temp = song_groups.iloc[train_val_idx]

X_test = X.iloc[test_idx]
y_test = y.iloc[test_idx]

# second split: separate validation set from remaining (15/85 ≈ 0.176 of train_temp)
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.17647, random_state=seed)
train_idx, val_idx = next(
    gss_val.split(X_train_temp, y_train_temp, groups=song_groups_train_temp)
)

X_train = X_train_temp.iloc[train_idx].reset_index(drop=True)
y_train = y_train_temp.iloc[train_idx].reset_index(drop=True)

X_val = X_train_temp.iloc[val_idx].reset_index(drop=True)
y_val = y_train_temp.iloc[val_idx].reset_index(drop=True)

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# diagnostic information
print("=== song-based split results ===")
print(f"train songs: {song_groups.iloc[train_val_idx].iloc[train_idx].nunique()}")
print(f"val songs: {song_groups.iloc[train_val_idx].iloc[val_idx].nunique()}")
print(f"test songs: {song_groups.iloc[test_idx].nunique()}")
print()
print(f"X_train shape: {X_train.shape} | class 1: {y_train.mean():.3f}")
print(f"X_val shape: {X_val.shape} | class 1: {y_val.mean():.3f}")
print(f"X_test shape: {X_test.shape} | class 1: {y_test.mean():.3f}")
print()

# verify no song overlap between splits
train_songs = set(song_groups.iloc[train_val_idx].iloc[train_idx].unique())
val_songs = set(song_groups.iloc[train_val_idx].iloc[val_idx].unique())
test_songs = set(song_groups.iloc[test_idx].unique())

assert len(train_songs & val_songs) == 0, "songs overlap between train and val"
assert len(train_songs & test_songs) == 0, "songs overlap between train and test"
assert len(val_songs & test_songs) == 0, "songs overlap between val and test"
print("verified: no song appears in multiple splits")

# export splits
X_train.to_csv("../../data/processed/X_train.csv", index=False)
X_val.to_csv("../../data/processed/X_val.csv", index=False)
X_test.to_csv("../../data/processed/X_test.csv", index=False)
y_train.to_csv("../../data/processed/y_train.csv", index=False)
y_val.to_csv("../../data/processed/y_val.csv", index=False)
y_test.to_csv("../../data/processed/y_test.csv", index=False)
