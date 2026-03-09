import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

INPUT_PATH = "outputs/player_surface_year.csv"
OUT_DIR = "outputs"
FIG_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

MIN_POINTS = 300
METRICS = ["serve_only_rate", "ace_rate", "avg_rally_shots"]
SURFACES = ["Hard", "Clay", "Grass"]


df = pd.read_csv(INPUT_PATH)
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year"])
df["Year"] = df["Year"].astype(int)

server = df[df["role"] == "server"].copy()
server["points"] = pd.to_numeric(server["points"], errors="coerce")
server = server.dropna(subset=["points"])
server = server[server["points"] >= MIN_POINTS].copy()

for m in METRICS:
    server[m] = pd.to_numeric(server[m], errors="coerce")


trend_rows = []
for metric in METRICS:
    tmp = server.dropna(subset=[metric]).copy()
    tmp["w"] = tmp["points"] * tmp[metric]

    tr = (
        tmp.groupby(["Year", "Surface"], as_index=False)
           .agg(points_sum=("points", "sum"), w_sum=("w", "sum"))
    )
    tr["value"] = tr["w_sum"] / tr["points_sum"]
    tr["metric"] = metric
    trend_rows.append(tr[["Year", "Surface", "points_sum", "value", "metric"]])

trend = pd.concat(trend_rows, ignore_index=True)
trend.to_csv(os.path.join(OUT_DIR, "trend_surface_year_weighted.csv"), index=False)


def run_wls(x, y, w):
    
    try:
        import statsmodels.api as sm
        X = sm.add_constant(x)
        model = sm.WLS(y, X, weights=w, missing="drop").fit()
        return {
            "intercept": float(model.params["const"]),
            "slope": float(model.params["Year"]),
            "slope_se": float(model.bse["Year"]),
            "slope_pvalue": float(model.pvalues["Year"]),
            "r2": float(model.rsquared),
            "n": int(model.nobs),
        }
    except Exception:
        # Manual weighted regression without SE/p-values
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        x = x[mask]; y = y[mask]; w = w[mask]
        n = len(x)
        if n < 3:
            return {"intercept": np.nan, "slope": np.nan, "slope_se": np.nan, "slope_pvalue": np.nan, "r2": np.nan, "n": n}

        xbar = np.average(x, weights=w)
        ybar = np.average(y, weights=w)
        cov_xy = np.average((x - xbar) * (y - ybar), weights=w)
        var_x = np.average((x - xbar) ** 2, weights=w)

        slope = cov_xy / var_x if var_x != 0 else np.nan
        intercept = ybar - slope * xbar

        yhat = intercept + slope * x
        ss_tot = np.sum(w * (y - ybar) ** 2)
        ss_res = np.sum(w * (y - yhat) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

        return {"intercept": float(intercept), "slope": float(slope), "slope_se": np.nan, "slope_pvalue": np.nan, "r2": float(r2), "n": int(n)}

results = []
for metric in METRICS:
    for surf in SURFACES:
        sub = trend[(trend["metric"] == metric) & (trend["Surface"] == surf)].copy()
        sub = sub.dropna(subset=["value", "points_sum"])
        if len(sub) < 5:
            continue

        out = run_wls(sub["Year"], sub["value"], sub["points_sum"])
        out["metric"] = metric
        out["surface"] = surf
        out["year_min"] = int(sub["Year"].min())
        out["year_max"] = int(sub["Year"].max())
        results.append(out)

res = pd.DataFrame(results).sort_values(["metric", "surface"])
res.to_csv(os.path.join(OUT_DIR, "trend_regression_results.csv"), index=False)

print("\n=== Trend regression results (value ~ Year), weighted by points ===")
print(res[["metric","surface","slope","slope_se","slope_pvalue","r2","n","year_min","year_max"]].to_string(index=False))


colors = {"Hard": "tab:blue", "Clay": "tab:orange", "Grass": "tab:green"}

for metric in METRICS:
    plt.figure()
    for surf in SURFACES:
        sub = trend[(trend["metric"] == metric) & (trend["Surface"] == surf)].copy()
        sub = sub.dropna(subset=["value", "points_sum"]).sort_values("Year")
        if len(sub) < 5:
            continue

        plt.plot(
            sub["Year"],
            sub["value"],
            label=surf,
            color=colors.get(surf, None)
        )

        rr = res[(res["metric"] == metric) & (res["surface"] == surf)]
        if len(rr) == 1 and np.isfinite(rr.iloc[0]["slope"]):
            a = rr.iloc[0]["intercept"]
            b = rr.iloc[0]["slope"]
            yhat = a + b * sub["Year"].values
            plt.plot(
                sub["Year"],
                yhat,
                linestyle="--",
                color=colors.get(surf, None)
            )

    plt.title(metric + ": weighted trend + fitted line")
    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "reg_trend_" + metric + ".png"), dpi=200)
    plt.close()

