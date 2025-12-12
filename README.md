This project builds a full end-to-end data science workflow for generating personalized music recommendations using Spotify playlist data. It integrates with the Spotify API to collect track metadata and audio features, merges these into a unified dataset, and applies unsupervised and supervised machine learning to identify sonic archetypes and predict user playlist preferences.

The final output is a prototype taste-scoring recommender system capable of ranking new tracks by their likelihood of matching a user's musical style.

Core Capabilities
Data Retrieval & Engineering

Automated playlist export using the Spotify Web API

Retrieval of both song metadata (songs.csv) and audio features (audio_features.csv)

Merged analytic file (tracks_merged.csv) containing all features needed for modeling

Cleaned, standardized dataset ready for clustering and classification

Machine Learning Pipeline
<img width="500" height="400" alt="correlation_heatmap" src="https://github.com/user-attachments/assets/f7b37284-87a8-48b5-b674-20ca8321a07c" />

Unsupervised Clustering: K-Means with multiple K values (6, 9, 12)
<img width="400" height="300" alt="pca_clusters" src="https://github.com/user-attachments/assets/bb122056-3d13-4a92-9e8a-039b96ff6998" />

Model Comparison: silhouette scoring + PCA visualization

Supervised Classification: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
<img width="500" height="300" alt="model_comparison" src="https://github.com/user-attachments/assets/b7a7127e-25bd-4fea-abf9-fd71f554ac40" />
Taste Scoring: Probability-based predictions ranking songs on a 0–1 “likelihood of playlist inclusion” scale

Interpretability: Feature importance, confusion matrices, ROC curves, and cross-model comparison
<img width="400" height="300" alt="roc_curves" src="https://github.com/user-attachments/assets/62ce1500-a1b1-484e-9893-b23a19d82ef7" />

Prototype Recommender

A future deployment could allow users to paste a Spotify playlist link, run the pipeline, and receive a ranked list of recommended tracks that match their learned taste profile.
MIT License

---

**Ideal for**: Music enthusiasts, data analysts, and portfolio projects showing Python scripting, API integration, and data engineering.
