import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_PATH = "outputs/player_surface_year.csv"
FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

MIN_POINTS = 300
MIN_POINTS_PER_SURFACE = 300
MIN_SURFACES = 2

df = pd.read_csv(INPUT_PATH)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

df = df[df["points"] >= MIN_POINTS].copy()

# helpers
def weighted_surface_year(df_role, metric):
    tmp = df_role.copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp["w"] = tmp["points"] * tmp[metric]
    out = (
        tmp.groupby(["Year", "Surface"], as_index=False)
           .agg(points_sum=("points", "sum"), w_sum=("w", "sum"))
    )
    out[metric + "_wavg"] = out["w_sum"] / out["points_sum"]
    return out[["Year", "Surface", metric + "_wavg"]]

def heatmap(metric_wavg_col, df_long, filename, title, vmin=None, vmax=None):
    piv = df_long.pivot(index="Surface", columns="Year", values=metric_wavg_col).reindex(["Hard","Clay","Grass"])
    plt.figure(figsize=(12, 3))
    plt.imshow(piv.values, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.yticks(range(piv.shape[0]), piv.index)
    step = max(1, piv.shape[1] // 10)
    plt.xticks(range(0, piv.shape[1], step), piv.columns[::step], rotation=45)
    plt.colorbar(label=metric_wavg_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    plt.close()

def zscore_columns(frame, cols):
    out = frame.copy()
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        mu = np.nanmean(x)
        sd = np.nanstd(x)
        if sd == 0 or np.isnan(sd):
            out[c + "_z"] = np.nan
        else:
            out[c + "_z"] = (x - mu) / sd
    return out

def specialization_distance(group, zcols):
    
    X = group[zcols].to_numpy(dtype=float)
    centroid = np.nanmean(X, axis=0)
    dists = np.sqrt(np.nansum((X - centroid) ** 2, axis=1))
    return pd.Series({
        "spec_mean_dist": float(np.nanmean(dists)),
        "spec_max_dist": float(np.nanmax(dists)),
        "n_surfaces": int(group["Surface"].nunique()),
        "total_points": float(group["points"].sum())
    })

def build_specialization(df_role, role_name, feat_cols):
   
    base = df_role.copy()

    for c in feat_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    base = base[base["points"] >= MIN_POINTS_PER_SURFACE].copy()

    surf_counts = base.groupby(["player","Year"])["Surface"].nunique().reset_index(name="n_surfaces")
    valid = surf_counts[surf_counts["n_surfaces"] >= MIN_SURFACES][["player","Year"]]
    base = base.merge(valid, on=["player","Year"], how="inner")

    base[feat_cols] = base.groupby(["player","Year"])[feat_cols].transform(lambda x: x.fillna(x.mean()))

    base_z = zscore_columns(base, feat_cols)
    zcols = [c + "_z" for c in feat_cols]

    spec = base_z.groupby(["player","Year"]).apply(lambda g: specialization_distance(g, zcols)).reset_index()
    spec = spec.rename(columns={
        "spec_mean_dist": f"specialization_{role_name}_mean_dist",
        "spec_max_dist": f"specialization_{role_name}_max_dist",
        "total_points": f"total_points_{role_name}",
        "n_surfaces": f"n_surfaces_{role_name}"
    })
    return spec


server = df[df["role"] == "server"].copy()
returner = df[df["role"] == "returner"].copy()

# heatmaps
serve_only_tr = weighted_surface_year(server, "serve_only_rate")
ace_tr = weighted_surface_year(server, "ace_rate")
rally_tr = weighted_surface_year(server, "avg_rally_shots")

heatmap("serve_only_rate_wavg", serve_only_tr, "heatmap_serve_only_rate.png",
        "Heatmap: Serve-only rate (server, weighted) by surface and year")
heatmap("ace_rate_wavg", ace_tr, "heatmap_ace_rate.png",
        "Heatmap: Ace rate (server, weighted) by surface and year")
heatmap("avg_rally_shots_wavg", rally_tr, "heatmap_avg_rally_shots.png",
        "Heatmap: Avg rally shots (server, weighted) by surface and year")

# new specialization with server and returner and standardized for each metric
server_feats = [
    "ace_rate", "serve_only_rate", "avg_rally_shots",
    "winner_rate", "ue_rate", "fe_rate",
    "sv_rate", "approach_rate",
    "pct_serve_4", "pct_serve_5", "pct_serve_6"
]

# Returner-side specialization uses different meaningful features
returner_feats = [
    "avg_rally_shots",
    "winner_rate", "ue_rate", "fe_rate",
    "approach_rate"
]

spec_server = build_specialization(server, "server", server_feats)
spec_returner = build_specialization(returner, "returner", returner_feats)

spec_both = spec_server.merge(spec_returner, on=["player","Year"], how="outer")

# Combined specialization 
spec_both["specialization_combined_mean_dist"] = spec_both[[
    "specialization_server_mean_dist",
    "specialization_returner_mean_dist"
]].mean(axis=1)

spec_both["specialization_combined_max_dist"] = spec_both[[
    "specialization_server_max_dist",
    "specialization_returner_max_dist"
]].max(axis=1)

spec_both.to_csv(os.path.join(OUT_DIR, "player_year_specialization_v2_both_roles.csv"), index=False)

spec_plot = spec_both.dropna(subset=["specialization_combined_mean_dist"]).copy()
spec_plot["decade"] = (spec_plot["Year"] // 10) * 10
decades = sorted(spec_plot["decade"].unique())
data = [spec_plot.loc[spec_plot["decade"] == d, "specialization_combined_mean_dist"].dropna().values for d in decades]

plt.figure(figsize=(12, 4))
plt.boxplot(data, labels=[str(d) for d in decades], showfliers=False)
plt.title("Specialization v2 (distance-based, standardized) by decade (combined server+returner)")
plt.xlabel("Decade")
plt.ylabel("Mean distance across surfaces")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "boxplot_specialization_v2_combined_by_decade.png"), dpi=200)
plt.close()

#scatter
tmp = server[server["points"] >= MIN_POINTS_PER_SURFACE].copy()
tmp["w_serv_only"] = tmp["points"] * tmp["serve_only_rate"]
py = tmp.groupby(["player","Year"], as_index=False).agg(points=("points","sum"), w_serv_only=("w_serv_only","sum"))
py["serve_only_py_wavg"] = py["w_serv_only"] / py["points"]

sc = spec_plot.merge(py[["player","Year","points","serve_only_py_wavg"]], on=["player","Year"], how="left")
sc = sc.dropna(subset=["serve_only_py_wavg"])

plt.figure(figsize=(6, 5))
plt.scatter(sc["serve_only_py_wavg"], sc["specialization_combined_mean_dist"],
            s=np.clip(sc["points"]/200, 5, 80), alpha=0.5)
plt.title("Player-year: specialization v2 (combined) vs serve-only rate")
plt.xlabel("Serve-only rate (player-year weighted avg, server)")
plt.ylabel("Specialization v2 (combined mean distance, z-scored features)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "scatter_specialization_v2_combined_vs_serve_only.png"), dpi=200)
plt.close()
