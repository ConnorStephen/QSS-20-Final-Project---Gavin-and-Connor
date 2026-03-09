import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


INPUT_SPECIALIZATION = "outputs/player_year_specialization_v2_both_roles.csv"
INPUT_PLAYER_SURFACE_YEAR = "outputs/player_surface_year.csv"

FIG_DIR = "figures"
OUT_DIR = "outputs"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


ERA_START = 2015
ERA_END = 2025

MIN_TOTAL_POINTS = 600

BUBBLE_MIN = 10
BUBBLE_MAX = 120


spec = pd.read_csv("outputs/player_year_specialization_v2_both_roles.csv")
spec["Year"] = pd.to_numeric(spec["Year"], errors="coerce")
spec = spec.dropna(subset=["Year"])
spec["Year"] = spec["Year"].astype(int)

if "specialization_combined_mean_dist" not in spec.columns:
    raise ValueError("Expected column specialization_combined_mean_dist in player_year_specialization_v2_both_roles.csv")

spec["total_points_server"] = pd.to_numeric(spec.get("total_points_server", np.nan), errors="coerce")
spec["total_points_returner"] = pd.to_numeric(spec.get("total_points_returner", np.nan), errors="coerce")
spec["total_points_combined"] = spec["total_points_server"].fillna(0) + spec["total_points_returner"].fillna(0)

spec["decade"] = (spec["Year"] // 10) * 10

spec_plot = spec.dropna(subset=["specialization_combined_mean_dist"]).copy()
spec_plot = spec_plot[spec_plot["total_points_combined"] >= MIN_TOTAL_POINTS].copy()


decades = sorted(spec_plot["decade"].unique())
data = []
ns = []

for d in decades:
    vals = spec_plot.loc[spec_plot["decade"] == d, "specialization_combined_mean_dist"].dropna().values
    data.append(vals)
    ns.append(len(vals))

plt.figure(figsize=(12, 4))
plt.boxplot(data, tick_labels=[str(d) for d in decades], showfliers=False)
plt.title("Specialization v2 (combined, standardized) by decade\n(with player-year counts)")
plt.xlabel("Decade")
plt.ylabel("Mean distance across surfaces (z-scored features)")
plt.xticks(rotation=45)

# annotate counts above each box
y_max = max([np.nanmax(v) if len(v) > 0 else 0 for v in data]) if len(data) > 0 else 1
for i, n in enumerate(ns, start=1):
    plt.text(i, y_max * 1.03, f"n={n}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "boxplot_specialization_v2_by_decade_with_counts.png"), dpi=200)
plt.close()

# Save decade counts table
decade_counts = pd.DataFrame({"decade": decades, "n_player_years": ns})
decade_counts.to_csv(os.path.join(OUT_DIR, "specialization_decade_counts.csv"), index=False)


psy = pd.read_csv(INPUT_PLAYER_SURFACE_YEAR)
psy["Year"] = pd.to_numeric(psy["Year"], errors="coerce")
psy = psy.dropna(subset=["Year"])
psy["Year"] = psy["Year"].astype(int)

psy = psy[psy["role"] == "server"].copy()
psy["points"] = pd.to_numeric(psy["points"], errors="coerce")
psy["serve_only_rate"] = pd.to_numeric(psy["serve_only_rate"], errors="coerce")

psy = psy.dropna(subset=["points", "serve_only_rate"])

psy["w_serv_only"] = psy["points"] * psy["serve_only_rate"]
py_serv = (
    psy.groupby(["player", "Year"], as_index=False)
       .agg(points_server=("points", "sum"), w_serv_only=("w_serv_only", "sum"))
)
py_serv["serve_only_py_wavg"] = py_serv["w_serv_only"] / py_serv["points_server"]

# merge with specialization
sc = spec_plot.merge(py_serv[["player", "Year", "points_server", "serve_only_py_wavg"]], on=["player", "Year"], how="left")
sc = sc.dropna(subset=["serve_only_py_wavg"]).copy()

sizes = sc["points_server"].fillna(0).to_numpy()
if sizes.max() > 0:
    sizes_scaled = BUBBLE_MIN + (BUBBLE_MAX - BUBBLE_MIN) * (sizes / sizes.max())
else:
    sizes_scaled = np.full(len(sc), BUBBLE_MIN)

decades_sc = sorted(sc["decade"].unique())
cmap = plt.get_cmap("tab10")

plt.figure(figsize=(7, 6))
for idx, d in enumerate(decades_sc):
    sub = sc[sc["decade"] == d]
    sub_sizes = sizes_scaled[sub.index.to_numpy() - sc.index.min()]
    plt.scatter(
        sub["serve_only_py_wavg"],
        sub["specialization_combined_mean_dist"],
        s=sub_sizes,
        alpha=0.55,
        label=str(d),
        color=cmap(idx % 10)
    )

plt.title("Player-year: specialization v2 (combined) vs serve-only rate\n(color = decade, size = server points)")
plt.xlabel("Serve-only rate (player-year weighted avg, server)")
plt.ylabel("Specialization v2 (combined mean distance)")
plt.legend(title="Decade", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "scatter_specialization_vs_serve_only_by_decade.png"), dpi=200)
plt.close()

# Top specialists vs most adaptable (2015–2025)

era = spec_plot[(spec_plot["Year"] >= ERA_START) & (spec_plot["Year"] <= ERA_END)].copy()


era["w_spec"] = era["total_points_combined"] * era["specialization_combined_mean_dist"]
player_era = (
    era.groupby("player", as_index=False)
       .agg(points=("total_points_combined", "sum"), w_spec=("w_spec", "sum"), n_years=("Year", "nunique"))
)
player_era["spec_wavg"] = player_era["w_spec"] / player_era["points"]

player_era = player_era[(player_era["points"] >= 2000) & (player_era["n_years"] >= 2)].copy()

top10 = player_era.sort_values("spec_wavg", ascending=False).head(10)
bottom10 = player_era.sort_values("spec_wavg", ascending=True).head(10)

# Bar chart: top specialists
plt.figure(figsize=(10, 4))
plt.bar(top10["player"], top10["spec_wavg"])
plt.title(f"Top 10 Surface Specialists (v2 combined), {ERA_START}–{ERA_END}")
plt.xlabel("Player")
plt.ylabel("Specialization (weighted avg across player-years)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"bar_top_specialists_{ERA_START}_{ERA_END}.png"), dpi=200)
plt.close()

# Bar chart: most adaptable (lowest specialization)
plt.figure(figsize=(10, 4))
plt.bar(bottom10["player"], bottom10["spec_wavg"])
plt.title(f"Top 10 Most Adaptable (lowest specialization), {ERA_START}–{ERA_END}")
plt.xlabel("Player")
plt.ylabel("Specialization (weighted avg across player-years)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, f"bar_top_adaptable_{ERA_START}_{ERA_END}.png"), dpi=200)
plt.close()

top10.to_csv(os.path.join(OUT_DIR, f"top_specialists_{ERA_START}_{ERA_END}.csv"), index=False)
bottom10.to_csv(os.path.join(OUT_DIR, f"top_adaptable_{ERA_START}_{ERA_END}.csv"), index=False)

