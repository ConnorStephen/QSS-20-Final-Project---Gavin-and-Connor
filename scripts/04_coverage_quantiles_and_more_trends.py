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
MIN_SURFACES_FOR_SPEC = 2

SPEC_FEATURES = ["ace_rate", "serve_only_rate", "avg_rally_shots"]

df = pd.read_csv(INPUT_PATH)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

server = df[df["role"] == "server"].copy()
server = server[server["points"] >= MIN_POINTS].copy()

# how much data per year and surface?
coverage_year_surface = (
    server.groupby(["Year", "Surface"], as_index=False)
          .agg(
              total_points=("points", "sum"),
              n_player_rows=("player", "count"),
              n_unique_players=("player", "nunique")
          )
          .sort_values(["Year", "Surface"])
)

coverage_year = (
    server.groupby("Year", as_index=False)
          .agg(
              total_points=("points", "sum"),
              n_player_rows=("player", "count"),
              n_unique_players=("player", "nunique")
          )
          .sort_values("Year")
)

coverage_year_surface.to_csv(os.path.join(OUT_DIR, "coverage_year_surface.csv"), index=False)
coverage_year.to_csv(os.path.join(OUT_DIR, "coverage_year.csv"), index=False)

# plotting total charted points over time by surface
plt.figure()
for surf in ["Hard", "Clay", "Grass"]:
    s = coverage_year_surface[coverage_year_surface["Surface"] == surf].sort_values("Year")
    plt.plot(s["Year"], s["total_points"], label=surf)
plt.title("Charted Points Over Time (Server Rows, MIN_POINTS filter)")
plt.xlabel("Year")
plt.ylabel("Total charted points (sum of points)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "coverage_total_points_by_surface.png"), dpi=200)
plt.close()

# plotting number of unique players over time (all surfaces combined)
plt.figure()
plt.plot(coverage_year["Year"], coverage_year["n_unique_players"])
plt.title("Unique Players Represented Over Time (Server Rows, MIN_POINTS filter)")
plt.xlabel("Year")
plt.ylabel("Number of unique players")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "coverage_unique_players_over_time.png"), dpi=200)
plt.close()

# Specialization distributions
spec_base = server[server["points"] >= MIN_POINTS_PER_SURFACE].copy()

surf_counts = (
    spec_base.groupby(["player", "Year"], as_index=False)["Surface"]
             .nunique()
             .rename(columns={"Surface": "n_surfaces"})
)
valid_player_year = surf_counts[surf_counts["n_surfaces"] >= MIN_SURFACES_FOR_SPEC][["player", "Year"]]
spec_base = spec_base.merge(valid_player_year, on=["player", "Year"], how="inner")

spec = (
    spec_base.groupby(["player", "Year"])[SPEC_FEATURES]
             .agg(lambda x: np.nanvar(x, ddof=0))
             .reset_index()
)
spec["specialization_index"] = spec[SPEC_FEATURES].mean(axis=1)
spec.to_csv(os.path.join(OUT_DIR, "player_year_specialization.csv"), index=False)

spec_year_stats = (
    spec.groupby("Year")["specialization_index"]
        .agg(
            mean="mean",
            median="median",
            p90=lambda x: np.nanpercentile(x, 90),
            n="count"
        )
        .reset_index()
        .sort_values("Year")
)
spec_year_stats.to_csv(os.path.join(OUT_DIR, "specialization_year_stats.csv"), index=False)

plt.figure()
plt.plot(spec_year_stats["Year"], spec_year_stats["mean"], label="Mean")
plt.plot(spec_year_stats["Year"], spec_year_stats["median"], label="Median")
plt.plot(spec_year_stats["Year"], spec_year_stats["p90"], label="90th percentile")
plt.title("Surface Specialization Over Time (Distribution Summary, Server Metrics)")
plt.xlabel("Year")
plt.ylabel("Specialization index")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "specialization_distribution_over_time.png"), dpi=200)
plt.close()

# More serve-dominance trends (weighted):

def weighted_trend(metric_name):
    tmp = server.copy()
    tmp["w_val"] = tmp["points"] * tmp[metric_name]
    out = (
        tmp.groupby(["Year", "Surface"], as_index=False)
           .agg(points_sum=("points", "sum"), w_sum=("w_val", "sum"))
    )
    out[metric_name + "_wavg"] = out["w_sum"] / out["points_sum"]
    return out[["Year", "Surface", metric_name + "_wavg"]]

trend_serve_only = weighted_trend("serve_only_rate")
trend_ace = weighted_trend("ace_rate")
trend_rally = weighted_trend("avg_rally_shots")

plt.figure()
for surf in ["Hard", "Clay", "Grass"]:
    s = trend_ace[trend_ace["Surface"] == surf].sort_values("Year")
    plt.plot(s["Year"], s["ace_rate_wavg"], label=surf)
plt.title("Ace Rate Over Time (Weighted by Points)")
plt.xlabel("Year")
plt.ylabel("Ace rate (weighted avg)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "ace_rate_trend_by_surface_all_years.png"), dpi=200)
plt.close()

plt.figure()
for surf in ["Hard", "Clay", "Grass"]:
    s = trend_rally[trend_rally["Surface"] == surf].sort_values("Year")
    plt.plot(s["Year"], s["avg_rally_shots_wavg"], label=surf)
plt.title("Average Rally Shots Over Time (Weighted by Points)")
plt.xlabel("Year")
plt.ylabel("Avg rally shots (weighted avg)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "avg_rally_shots_trend_by_surface_all_years.png"), dpi=200)
plt.close()

trend_serve_only.to_csv(os.path.join(OUT_DIR, "trend_serve_only_rate.csv"), index=False)
trend_ace.to_csv(os.path.join(OUT_DIR, "trend_ace_rate.csv"), index=False)
trend_rally.to_csv(os.path.join(OUT_DIR, "trend_avg_rally_shots.csv"), index=False)