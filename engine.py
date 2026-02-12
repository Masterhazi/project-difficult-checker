# engine.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from typing import Dict

MODEL_NAME = "all-MiniLM-L6-v2"

class ProjectDifficultyEngine:
    def __init__(self, artifacts_dir="models"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.centroids = np.load(f"{artifacts_dir}/centroids.npy")
        with open(f"{artifacts_dir}/weights.json") as f: self.weights = json.load(f)
        with open(f"{artifacts_dir}/thresholds.json") as f: self.thresholds = json.load(f)
        with open(f"{artifacts_dir}/cluster_to_tier.json") as f: self.cluster_to_tier = json.load(f)
        self.label_map = {0:"Easy", 1:"Average", 2:"Difficult"}

    def embed(self, combined_text: str):
        emb = self.model.encode([combined_text], normalize_embeddings=True)  # shape (1, dim)
        return emb[0]

    def nearest_cluster(self, emb_vec):
        # centroids are normalized; emb_vec normalized already
        sims = self.centroids.dot(emb_vec)  # cosine similarity (since normalized)
        return int(np.argmax(sims)), float(np.max(sims))

    def structural_weighted_score(self, feature_dict: Dict[str,int]):
        score = 0.0
        for k,v in self.weights.items():
            score += feature_dict.get(k, 0) * float(v)
        return score

    def mp_num_from_score(self, score: float):
        q1 = float(self.thresholds["q1"])
        q2 = float(self.thresholds["q2"])
        if score <= q1:
            return 0
        elif score <= q2:
            return 1
        else:
            return 2

    def semantic_num_from_cluster(self, cluster_id: int):
        return int(self.cluster_to_tier[str(cluster_id)])

    def ensemble(self, mp_num, semantic_num, alpha=0.6):
        return int(round(alpha*mp_num + (1-alpha)*semantic_num))

    def predict(self, title, abstract, feature_dict, alpha=0.6):
        combined_text = (title or "") + " " + (abstract or "")
        emb = self.embed(combined_text)
        cluster_id, cluster_sim = self.nearest_cluster(emb)
        semantic_num = self.semantic_num_from_cluster(cluster_id)

        weighted_score = self.structural_weighted_score(feature_dict)
        mp_num = self.mp_num_from_score(weighted_score)

        final_num = self.ensemble(mp_num, semantic_num, alpha=alpha)
        confidence = 1 - abs(mp_num - semantic_num)/2.0

        return {
            "Final_Label": self.label_map[final_num],
            "Final_num": final_num,
            "MP_num": mp_num,
            "Semantic_num": semantic_num,
            "Weighted_Score": weighted_score,
            "Cluster": cluster_id,
            "Cluster_similarity": cluster_sim,
            "Confidence": round(confidence, 3)
        }
