from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
EDA_DIR = ROOT / "eda_plots"
SONGS_PATH = ROOT / "songs.csv"
FEATURES_PATH = ROOT / "audio_features.csv"


def run_eda_and_merge() -> pd.DataFrame:
    songs_df = pd.read_csv(SONGS_PATH)
    features_df = pd.read_csv(FEATURES_PATH)

    print(f"songs.csv shape: {songs_df.shape}")
    print(f"songs.csv columns: {list(songs_df.columns)}")
    print(f"audio_features.csv shape: {features_df.shape}")
    print(f"audio_features.csv columns: {list(features_df.columns)}")

    EDA_DIR.mkdir(parents=True, exist_ok=True)

    candidate_keys = ["id", "track_id", "song_id"]
    lower_songs = {col.lower(): col for col in songs_df.columns}
    lower_features = {col.lower(): col for col in features_df.columns}

    merge_kwargs = {"how": "inner"}

    if "id" in lower_songs and "track_id" in lower_features:
        merge_kwargs["left_on"] = lower_songs["id"]
        merge_kwargs["right_on"] = lower_features["track_id"]
        print(
            f'Using explicit join: songs.{merge_kwargs["left_on"]} <-> '
            f'audio_features.{merge_kwargs["right_on"]}'
        )
    else:
        join_key = None
        for key in candidate_keys:
            if key in lower_songs and key in lower_features:
                join_key = lower_songs[key]
                feature_key = lower_features[key]
                if feature_key != join_key:
                    features_df = features_df.rename(columns={feature_key: join_key})
                break

        if join_key is None:
            raise ValueError("No common join key found among id, track_id, or song_id.")

        print(f"Using join key: {join_key}")
        merge_kwargs["on"] = join_key

    merged_df = pd.merge(songs_df, features_df, **merge_kwargs)
    print(f"Merged shape: {merged_df.shape}")
    print("Missing values per column:")
    print(merged_df.isna().sum())

    output_path = ROOT / "tracks_merged.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

    numeric_cols = merged_df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        print("No numeric columns found in merged data for EDA.")
    else:
        print("Numeric columns:", list(numeric_cols))
        print(merged_df[numeric_cols].describe().T)

        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(merged_df[col].dropna(), kde=False)
            plt.title(f"{col} distribution")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(EDA_DIR / f"{col}_hist.png")
            plt.close()

        corr = merged_df[numeric_cols].corr()
        print("Correlation table:")
        print(corr)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(EDA_DIR / "correlation_heatmap.png")
        plt.close()

    categorical_cols = merged_df.select_dtypes(exclude="number").columns
    if len(categorical_cols) == 0:
        print("No categorical columns found in merged data for EDA.")
    else:
        print("Categorical columns:", list(categorical_cols))
        for col in categorical_cols:
            value_counts = merged_df[col].value_counts(dropna=False)
            unique_count = value_counts.shape[0]
            if unique_count > 10:
                print(f"Top categories for {col} (10 of {unique_count}):")
                print(value_counts.head(10))
            else:
                print(f"Value counts for {col}:")
                print(value_counts)

    return merged_df
