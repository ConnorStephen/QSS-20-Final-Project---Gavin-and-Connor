import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

INPUT = "outputs/player_year_style_fingerprint.csv"
OUT_DIR = "outputs"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

K_RANGE = range(3, 9)
RANDOM_STATE = 42

df = pd.read_csv(INPUT)

feat_cols = [c for c in df.columns if c not in ["player","Year","role"]]
wide = df.pivot_table(index=["player","Year"], columns="role", values=feat_cols)

wide.columns = [f"{a}_{b}" for (a,b) in wide.columns]
wide = wide.reset_index()

X = wide.drop(columns=["player","Year"]).apply(pd.to_numeric, errors="coerce").dropna()
wide = wide.loc[X.index].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.values)

sil = []
best_k = None
best_s = -1

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    lab = km.fit_predict(X_scaled)
    s = silhouette_score(X_scaled, lab)
    sil.append((k, s))
    if s > best_s:
        best_s = s
        best_k = k

sil_df = pd.DataFrame(sil, columns=["k","silhouette"])
sil_df.to_csv(os.path.join(OUT_DIR, "style_silhouette_by_k.csv"), index=False)

plt.figure()
plt.plot(sil_df["k"], sil_df["silhouette"], marker="o")
plt.title("Silhouette by k (style fingerprints)")
plt.xlabel("k")
plt.ylabel("Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "style_silhouette_by_k.png"), dpi=200)
plt.close()

# final clustering
km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=50)
wide["cluster"] = km.fit_predict(X_scaled)

wide.to_csv(os.path.join(OUT_DIR, "player_year_style_clusters.csv"), index=False)

# PCA plot
pca = PCA(n_components=2, random_state=RANDOM_STATE)
Z = pca.fit_transform(X_scaled)

plt.figure(figsize=(7,6))
plt.scatter(Z[:,0], Z[:,1], c=wide["cluster"], alpha=0.6)
plt.title(f"PCA: style fingerprint clusters (k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_style_clusters.png"), dpi=200)
plt.close()

print("Best k:", best_k, "silhouette:", best_s)
