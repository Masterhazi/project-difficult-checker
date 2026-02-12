# train.py
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

# ---- config (from your notebook)
K = 6
weights = {
    "Deep Learning": 3,
    "Medical/Health": 3,
    "Real-Time": 2,
    "Vision": 2,
    "Finance/Fraud": 2,
    "Time-Series": 2,
    "Low Data": 2,
    "NLP": 1.5,
    "Unsupervised": 1.5,
    "Noisy Data": 1
}

# load CSV (path to your sheet)
df = pd.read_csv("data/Mini-Projects - Sheet1.csv", encoding="cp1252")
df.dropna(subset=["Project Title","Justification"], inplace=True)
df["Combined_Text"] = df["Project Title"].astype(str) + " " + df["Justification"].astype(str)

# embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(df["Combined_Text"].tolist(), normalize_embeddings=True)
emb_norm = emb  # already normalized because we used normalize_embeddings=True

# fit agglomerative
agg = AgglomerativeClustering(n_clusters=K, metric='cosine', linkage='average')
labels = agg.fit_predict(emb_norm)
df["Semantic_Cluster"] = labels

# compute Weighted_Score
feat_cols = list(weights.keys())
df["Weighted_Score"] = sum(df[col]*w for col,w in weights.items())

# derive MP multiclass via quantiles (q1,q2)
q1 = df["Weighted_Score"].quantile(0.33)
q2 = df["Weighted_Score"].quantile(0.66)
def mp_multiclass(score):
    if score <= q1:
        return 0
    elif score <= q2:
        return 1
    else:
        return 2
df["MP_num"] = df["Weighted_Score"].apply(mp_multiclass)

# compute cluster weighted means (for mapping clusters->difficulty)
cluster_weighted = df.groupby("Semantic_Cluster")["Weighted_Score"].mean().sort_values()
# choose mapping (you used:)
easy_clusters = [5,4]
avg_clusters = [0]
hard_clusters = [2,1,3]
cluster_to_tier = {}
for c in easy_clusters:
    cluster_to_tier[int(c)] = 0
for c in avg_clusters:
    cluster_to_tier[int(c)] = 1
for c in hard_clusters:
    cluster_to_tier[int(c)] = 2
df["Semantic_num"] = df["Semantic_Cluster"].map(cluster_to_tier)

# ensemble (same formula you used)
df["Ensemble_score"] = 0.6*df["MP_num"] + 0.4*df["Semantic_num"]
df["Final_num"] = df["Ensemble_score"].round().astype(int)
reverse_map = {0:"Easy", 1:"Average", 2:"Difficult"}
df["Final_Output"] = df["Final_num"].map(reverse_map)

# Save artifacts:
# 1) cluster centroids (mean normalized embedding per cluster)
centroids = np.vstack([emb_norm[labels==c].mean(axis=0) for c in range(K)])
centroids = normalize(centroids)  # ensure normalized centroids
np.save("models/centroids.npy", centroids)

# 2) weights, thresholds, mapping
json.dump(weights, open("models/weights.json","w"))
json.dump({"q1":float(q1), "q2":float(q2)}, open("models/thresholds.json","w"))
json.dump(cluster_to_tier, open("models/cluster_to_tier.json","w"))

# Save example results for audit
df.to_csv("models/training_results.csv", index=False)
print("Training finished. Artifacts saved in models/")
