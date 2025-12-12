from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent


def run_split_and_create_template(
    input_path: str = "tracks_with_clusters.csv",
    train_frac: float = 0.6,
    test_frac: float = 0.2,
    hidden_frac: float = 0.2,
    random_state: int = 42,
) -> None:
    if not abs(train_frac + test_frac + hidden_frac - 1.0) < 1e-6:
        raise ValueError("train_frac, test_frac, and hidden_frac must sum to 1.")

    input_file = ROOT / input_path
    df = pd.read_csv(input_file)

    # First split off the hidden set
    temp_df, hidden_df = train_test_split(
        df, test_size=hidden_frac, random_state=random_state, shuffle=True
    )

    # Split the remaining data into train and test based on remaining proportions
    adjusted_test_frac = test_frac / (train_frac + test_frac)
    train_df, test_df = train_test_split(
        temp_df, test_size=adjusted_test_frac, random_state=random_state, shuffle=True
    )

    train_df = train_df.copy()
    test_df = test_df.copy()
    hidden_df = hidden_df.copy()

    train_df["split"] = "train"
    test_df["split"] = "test"
    hidden_df["split"] = "hidden"

    train_path = ROOT / "tracks_train.csv"
    test_path = ROOT / "tracks_test.csv"
    hidden_path = ROOT / "tracks_hidden.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    hidden_df.to_csv(hidden_path, index=False)

    # Labeling template from training set
    template_df = train_df.copy()
    template_df["would_add_to_playlist"] = pd.NA
    template_path = ROOT / "tracks_with_semantic_template.csv"
    template_df.to_csv(template_path, index=False)

    print("Data split summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Train rows: {len(train_df)} -> {train_path}")
    print(f"  Test rows: {len(test_df)} -> {test_path}")
    print(f"  Hidden rows: {len(hidden_df)} -> {hidden_path}")
    print(f"Labeling template saved to: {template_path}")
