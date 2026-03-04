import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

INPUT_PATH = "outputs/player_surface_year.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

MIN_POINTS = 300                 # minimum points for including a (player, year, surface, role) row
MIN_POINTS_PER_SURFACE = 300     # minimum points per surface when computing specialization
MIN_SURFACES_FOR_SPEC = 2        # require at least 2 surfaces for specialization

SPEC_FEATURES = ["ace_rate", "serve_only_rate", "avg_rally_shots"]

df = pd.read_csv(INPUT_PATH)

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

server = df[df["role"] == "server"].copy()

server = server[server["points"] >= MIN_POINTS].copy()

# Serve-only rate trend by surface 
# Weighted average across players by year+surface:
# sum(points * serve_only_rate) / sum(points)

tmp = server.copy()
tmp["w_serv_only"] = tmp["points"] * tmp["serve_only_rate"]

trend = (
    tmp.groupby(["Year", "Surface"], as_index=False)
       .agg(points_sum=("points", "sum"), w_serv_only_sum=("w_serv_only", "sum"))
)

trend["serve_only_rate_wavg"] = trend["w_serv_only_sum"] / trend["points_sum"]

plt.figure()
for surf in ["Hard", "Clay", "Grass"]:
    s = trend[trend["Surface"] == surf].sort_values("Year")
    plt.plot(s["Year"], s["serve_only_rate_wavg"], label=surf)

plt.title("Serve-Only Rate Over Time (Weighted by Points)")
plt.xlabel("Year")
plt.ylabel("Serve-only rate (weighted avg)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "serve_only_rate_trend_by_surface_all_years.png"), dpi=200)
plt.close()

# Specialization index (player-year) 
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

spec_year = (
    spec.groupby("Year", as_index=False)["specialization_index"]
        .mean()
        .dropna()
        .sort_values("Year")
)

plt.figure()
plt.plot(spec_year["Year"], spec_year["specialization_index"])
plt.title("Average Surface Specialization Over Time (Server Metrics)")
plt.xlabel("Year")
plt.ylabel("Specialization index (avg variance across surfaces)")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "specialization_over_time_filtered.png"), dpi=200)
plt.close()

spec.to_csv("outputs/player_year_specialization.csv", index=False)

print("Server rows after MIN_POINTS filter:", len(server))
print("Trend rows:", len(trend))
print("Spec rows (player-year after surface/points filters):", len(spec))
print("Years in trend:", int(trend["Year"].min()), "-", int(trend["Year"].max()))
print("Years in spec:", int(spec["Year"].min()), "-", int(spec["Year"].max()))