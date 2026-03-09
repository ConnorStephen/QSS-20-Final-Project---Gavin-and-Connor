import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT = "outputs/player_year_style_clusters.csv"
PLAYER_SURFACE_YEAR = "outputs/player_surface_year.csv"

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MIN_PLAYER_YEARS_PER_YEAR_FLOOR = 10     
MIN_PLAYER_YEARS_PER_YEAR_FRAC = 0.25    

MIN_POINTS_SURFACE_FOR_MAIN = 200

SURFACES = ["Hard", "Clay", "Grass"]


df = pd.read_csv(INPUT)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)


feat_cols = [c for c in df.columns if c not in ["player", "Year", "cluster"] and not c.startswith("points_")]
for c in feat_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")


counts = df.groupby(["Year", "cluster"]).size().reset_index(name="n")
tot = df.groupby("Year").size().reset_index(name="n_total")
counts = counts.merge(tot, on="Year", how="left")
counts["share"] = counts["n"] / counts["n_total"]

median_n = float(tot["n_total"].median()) if len(tot) > 0 else 0
min_needed = int(max(MIN_PLAYER_YEARS_PER_YEAR_FLOOR, np.floor(MIN_PLAYER_YEARS_PER_YEAR_FRAC * median_n)))

good_years = tot[tot["n_total"] >= min_needed]["Year"]
counts_f = counts[counts["Year"].isin(good_years)].copy()

counts_f.to_csv(os.path.join(OUT_DIR, "style_cluster_prevalence_filtered.csv"), index=False)

pivot = counts_f.pivot(index="Year", columns="cluster", values="share").fillna(0).sort_index()

plt.figure(figsize=(10, 4))
plt.stackplot(
    pivot.index,
    [pivot[c].values for c in pivot.columns],
    labels=[f"Cluster {c}" for c in pivot.columns]
)
plt.title(f"Playstyle cluster prevalence over time (years with n >= {min_needed})")
plt.xlabel("Year")
plt.ylabel("Share of player-years")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "style_cluster_share_over_time_filtered.png"), dpi=200)
plt.close()


centroids = df.groupby("cluster")[feat_cols].mean()
centroids.to_csv(os.path.join(OUT_DIR, "style_cluster_centroids_no_points.csv"))

cent_z = (centroids - centroids.mean(axis=0)) / centroids.std(axis=0)

plt.figure(figsize=(12, 3 + 0.5 * len(cent_z)))
plt.imshow(cent_z.values, aspect="auto", interpolation="nearest")
plt.yticks(range(len(cent_z.index)), [f"Cluster {c}" for c in cent_z.index])
plt.xticks(range(len(cent_z.columns)), cent_z.columns, rotation=90)
plt.colorbar(label="Centroid z-score (across clusters)")
plt.title("Playstyle cluster centroids (standardized; points excluded)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "style_cluster_centroid_heatmap_no_points.png"), dpi=200)
plt.close()


psy = pd.read_csv(PLAYER_SURFACE_YEAR)
psy["Year"] = pd.to_numeric(psy["Year"], errors="coerce")
psy = psy.dropna(subset=["Year"])
psy["Year"] = psy["Year"].astype(int)

psy["points"] = pd.to_numeric(psy["points"], errors="coerce")
psy = psy.dropna(subset=["points"])

# total points per player-year-surface across roles
surf_pts = psy.groupby(["player", "Year", "Surface"], as_index=False).agg(points=("points", "sum"))
surf_pts = surf_pts[surf_pts["Surface"].isin(SURFACES)].copy()

# merge clusters onto surface rows (so each surface row knows its cluster)
surf_pts = surf_pts.merge(df[["player", "Year", "cluster"]], on=["player", "Year"], how="inner")

main = surf_pts.sort_values(["player", "Year", "points"], ascending=[True, True, False]).drop_duplicates(["player", "Year"])
main = main[main["points"] >= MIN_POINTS_SURFACE_FOR_MAIN].copy()
main = main.rename(columns={"Surface": "main_surface", "points": "main_surface_points"})

by_main = (
    main.groupby(["main_surface", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "n_player_years"})
)

tot_main = by_main.groupby("main_surface")["n_player_years"].sum().reset_index(name="n_total")
by_main = by_main.merge(tot_main, on="main_surface", how="left")
by_main["share"] = by_main["n_player_years"] / by_main["n_total"]
by_main.to_csv(os.path.join(OUT_DIR, "style_cluster_share_by_main_surface.csv"), index=False)

piv_main = by_main.pivot(index="main_surface", columns="cluster", values="share").fillna(0).reindex(SURFACES)

CLUSTER_NAMES = {
    0: "Return-First / Defensive Rally",
    1: "Serve-and-Volley / Net Rush",
    2: "First-Strike / Serve-Dominant",
    3: "Baseline Grinder / Rally-Heavy",
    4: "All-Court / Net-Approach",
}

cluster_cols = list(pivot.columns)
labels = [CLUSTER_NAMES.get(int(c), f"Cluster {c}") for c in cluster_cols]

plt.figure(figsize=(10, 4))
plt.stackplot(pivot.index, [pivot[c].values for c in cluster_cols], labels=labels)
plt.title("Playstyle prevalence over time")
plt.xlabel("Year")
plt.ylabel("Share of player-years")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "style_cluster_share_over_time_named.png"), dpi=200)
plt.close()

tot_py = surf_pts.groupby(["player", "Year"], as_index=False).agg(py_points=("points", "sum"))
mix = surf_pts.merge(tot_py, on=["player", "Year"], how="left")
mix["surface_share"] = mix["points"] / mix["py_points"]

mix_agg = (
    mix.groupby(["Surface", "cluster"], as_index=False)
       .agg(weighted_player_years=("surface_share", "sum"))
)

tot_mix = mix_agg.groupby("Surface")["weighted_player_years"].sum().reset_index(name="total_weight")
mix_agg = mix_agg.merge(tot_mix, on="Surface", how="left")
mix_agg["share"] = mix_agg["weighted_player_years"] / mix_agg["total_weight"]
mix_agg.to_csv(os.path.join(OUT_DIR, "style_cluster_share_by_surface_mix.csv"), index=False)

piv_mix = mix_agg.pivot(index="Surface", columns="cluster", values="share").fillna(0).reindex(SURFACES)

cluster_cols = sorted(piv_mix.columns)
labels = [CLUSTER_NAMES.get(int(c), f"Cluster {c}") for c in cluster_cols]

bottom = np.zeros(len(piv_mix.index))
plt.figure(figsize=(7, 4))
for cl, lab in zip(cluster_cols, labels):
    vals = piv_mix[cl].values
    plt.bar(piv_mix.index, vals, bottom=bottom, label=lab)
    bottom = bottom + vals

plt.title("Playstyle composition by surface (fractional weighting)")
plt.xlabel("Surface")
plt.ylabel("Share of weighted player-years")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "style_cluster_share_by_surface_named.png"), dpi=200)
plt.close()

