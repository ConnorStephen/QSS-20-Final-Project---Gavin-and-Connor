import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT = "outputs/player_year_style_clusters.csv"
OUT_DIR = "outputs"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MIN_PLAYER_YEARS_PER_YEAR = 40

N_EXEMPLARS = 15

df = pd.read_csv(INPUT)

feat_cols = [c for c in df.columns if c not in ["player", "Year", "cluster"]]

for c in feat_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# cluster centroids
centroids = df.groupby("cluster")[feat_cols].mean().reset_index()
centroids.to_csv(os.path.join(OUT_DIR, "style_cluster_centroids.csv"), index=False)

# deviations from means (cluster distinguishing)
overall = df[feat_cols].mean()
dev = centroids.copy()
for c in feat_cols:
    dev[c] = dev[c] - overall[c]
dev.to_csv(os.path.join(OUT_DIR, "style_cluster_deviation_from_mean.csv"), index=False)

# “top features” per cluster (absolute deviation)
top_rows = []
for cl in sorted(df["cluster"].unique()):
    row = dev[dev["cluster"] == cl].iloc[0]
    diffs = []
    for c in feat_cols:
        diffs.append((c, float(row[c])))
    diffs.sort(key=lambda x: abs(x[1]), reverse=True)
    for rank, (c, val) in enumerate(diffs[:12], start=1):
        top_rows.append({"cluster": cl, "rank": rank, "feature": c, "deviation": val})
top_feat = pd.DataFrame(top_rows)
top_feat.to_csv(os.path.join(OUT_DIR, "style_cluster_top_features.csv"), index=False)

# examples
exemplar_list = []
for cl in sorted(df["cluster"].unique()):
    centroid = df[df["cluster"] == cl][feat_cols].mean().values
    X = df[feat_cols].values
    dist = np.sqrt(np.nansum((X - centroid) ** 2, axis=1))

    tmp = df[["player", "Year", "cluster"]].copy()
    tmp["dist_to_centroid"] = dist
    ex = tmp[tmp["cluster"] == cl].sort_values("dist_to_centroid").head(N_EXEMPLARS)
    exemplar_list.append(ex)

exemplars = pd.concat(exemplar_list, ignore_index=True)
exemplars.to_csv(os.path.join(OUT_DIR, "style_cluster_exemplars.csv"), index=False)

#heatmap centroids
cent_mat = centroids.set_index("cluster")[feat_cols]
cent_z = (cent_mat - cent_mat.mean(axis=0)) / cent_mat.std(axis=0)

plt.figure(figsize=(12, 3 + 0.5 * len(cent_z)))
plt.imshow(cent_z.values, aspect="auto", interpolation="nearest")
plt.yticks(range(len(cent_z.index)), [f"Cluster {c}" for c in cent_z.index])
plt.xticks(range(len(cent_z.columns)), cent_z.columns, rotation=90)
plt.colorbar(label="Centroid z-score (across clusters)")
plt.title("Style cluster centroids (standardized across clusters)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "style_cluster_centroid_heatmap.png"), dpi=200)
plt.close()


counts = df.groupby(["Year", "cluster"]).size().reset_index(name="n")
tot = df.groupby("Year").size().reset_index(name="n_total")
counts = counts.merge(tot, on="Year", how="left")
counts["share"] = counts["n"] / counts["n_total"]

good_years = tot[tot["n_total"] >= MIN_PLAYER_YEARS_PER_YEAR]["Year"]
counts = counts[counts["Year"].isin(good_years)].copy()

pivot = counts.pivot(index="Year", columns="cluster", values="share").fillna(0).sort_index()

plt.figure(figsize=(10, 4))
plt.stackplot(pivot.index, [pivot[c].values for c in pivot.columns], labels=[f"Cluster {c}" for c in pivot.columns])
plt.title(f"Cluster prevalence over time (years with n >= {MIN_PLAYER_YEARS_PER_YEAR} player-years)")
plt.xlabel("Year")
plt.ylabel("Share of player-years")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "style_cluster_share_over_time_filtered.png"), dpi=200)
plt.close()

counts.to_csv(os.path.join(OUT_DIR, "style_cluster_prevalence_filtered.csv"), index=False)

