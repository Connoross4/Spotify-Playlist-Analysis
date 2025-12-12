from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
MERGED_PATH = ROOT / "tracks_merged.csv"
CLUSTER_PLOTS_DIR = ROOT / "cluster_plots"


def summarize_clusters(df: pd.DataFrame, feature_cols, label_col: str = "cluster_label") -> pd.DataFrame:
    clusters_df = df.dropna(subset=[label_col])
    cluster_means = clusters_df.groupby(label_col)[feature_cols].mean()

    overall_mean = df[feature_cols].mean()
    overall_std = df[feature_cols].std()

    z_scores = (cluster_means - overall_mean) / overall_std

    CLUSTER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    means_path = CLUSTER_PLOTS_DIR / "cluster_centroids_means.csv"
    zscores_path = CLUSTER_PLOTS_DIR / "cluster_centroids_zscores.csv"

    cluster_means.T.reset_index().rename(columns={"index": "feature"}).to_csv(means_path, index=False)
    z_scores.T.reset_index().rename(columns={"index": "feature"}).to_csv(zscores_path, index=False)

    print(f"Saved cluster centroid means to {means_path}")
    print(f"Saved cluster centroid z-scores to {zscores_path}")

    def describe_z(z: float) -> str:
        if z >= 1.5:
            return "very high"
        if 1.0 <= z < 1.5:
            return "high"
        if -1.5 < z <= -1.0:
            return "low"
        if z <= -1.5:
            return "very low"
        return "near average"

    for cluster_label in cluster_means.index:
        z_row = z_scores.loc[cluster_label]
        top_features = z_row.abs().sort_values(ascending=False).head(3)
        print(f"Cluster {cluster_label}:")
        for feature in top_features.index:
            z_val = z_row[feature]
            mean_val = cluster_means.loc[cluster_label, feature]
            descriptor = describe_z(z_val)
            sign = "+" if z_val >= 0 else "-"
            print(f"  {feature} is {descriptor} (mean = {mean_val:.2f}, z = {sign}{abs(z_val):.2f})")
    return cluster_means


def run_clustering() -> pd.DataFrame:
    df = pd.read_csv(MERGED_PATH)

    numeric_features = [
        "duration_ms",
        "popularity",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
    ]

    missing_features = [col for col in numeric_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required numeric features in merged data: {missing_features}")

    feature_data = df[numeric_features]
    clean_data = feature_data.dropna()
    if clean_data.shape[0] < feature_data.shape[0]:
        dropped = feature_data.shape[0] - clean_data.shape[0]
        print(f"Dropped {dropped} rows with missing numeric features for clustering.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(clean_data)

    ks = [6, 9, 12]
    silhouette_scores = {}
    best_k = None
    best_score = -1
    best_model = None
    best_labels = None

    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)
        score = silhouette_score(scaled_features, labels)
        silhouette_scores[k] = score
        print(f"Silhouette score for k={k}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k
            best_model = model
            best_labels = labels

    print(f"Selected k={best_k} with silhouette score {best_score:.4f}")

    df["cluster_label"] = pd.NA
    df.loc[clean_data.index, "cluster_label"] = best_labels

    output_path = ROOT / "tracks_with_clusters.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved clustered data to {output_path}")

    CLUSTER_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Silhouette score comparison chart
    plt.figure(figsize=(8, 4))
    plt.bar(list(silhouette_scores.keys()), list(silhouette_scores.values()), color="skyblue")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores by k")
    plt.tight_layout()
    plt.savefig(CLUSTER_PLOTS_DIR / "silhouette_scores.png")
    plt.close()

    # PCA visualization
    pca = PCA(n_components=2, random_state=42)
    pca_components = pca.fit_transform(scaled_features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_components[:, 0],
        pca_components[:, 1],
        c=best_labels,
        cmap="tab10",
        alpha=0.7,
        edgecolor="k",
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(f"PCA Clusters (k={best_k})")
    plt.legend(*scatter.legend_elements(), title="Cluster")
    plt.tight_layout()
    plt.savefig(CLUSTER_PLOTS_DIR / "pca_clusters.png")
    plt.close()

    summarize_clusters(df, numeric_features, label_col="cluster_label")

    return df
