import pandas as pd
import numpy as np
import os

INPUT_PATH = "data/processed/points_all_parsed.csv"   
OUT_PATH = "outputs/player_year_style_fingerprint.csv"
os.makedirs("outputs", exist_ok=True)

CHUNKSIZE = 400000
MIN_POINTS_PLAYER_YEAR = 200  # changed to smaller or else not a lot of data

# Define last-shot categories based on the match chart key
NET_SHOTS = set(["v","z","h","i","j","k","o","p"])   # volleys/halfvolleys/swinging/smashes
SLICE_SHOTS = set(["r","s"])                         # slices
DROP_SHOTS = set(["u","y"])                          # drops
GROUND_SHOTS = set(["f","b"])                        # groundstrokes

def role_split(df):
    
    out = []

    p1_serv = df[df["Svr"] == 1].copy()
    p1_serv["player"] = p1_serv["Player 1"]
    p1_serv["role"] = "server"
    out.append(p1_serv)

    p2_serv = df[df["Svr"] == 2].copy()
    p2_serv["player"] = p2_serv["Player 2"]
    p2_serv["role"] = "server"
    out.append(p2_serv)

    p1_ret = df[df["Svr"] == 2].copy()
    p1_ret["player"] = p1_ret["Player 1"]
    p1_ret["role"] = "returner"
    out.append(p1_ret)

    p2_ret = df[df["Svr"] == 1].copy()
    p2_ret["player"] = p2_ret["Player 2"]
    p2_ret["role"] = "returner"
    out.append(p2_ret)

    return pd.concat(out, ignore_index=True)

def summarize_style(g):
    n = len(g)
    if n == 0:
        return pd.Series()

    rs = pd.to_numeric(g["rally_shots"], errors="coerce").fillna(0)

    last = g["last_shot_type"].astype(str).str.strip().replace({"nan": ""})
    last_net = last.isin(NET_SHOTS).sum()
    last_slice = last.isin(SLICE_SHOTS).sum()
    last_drop = last.isin(DROP_SHOTS).sum()
    last_ground = last.isin(GROUND_SHOTS).sum()

    short_pts = (rs <= 2).sum()
    long_pts = (rs >= 6).sum()

    serve_only = (g["point_end"] == "serve_only").sum()

    return pd.Series({
        "points": n,
        "serve_only_rate": serve_only / n,
        "sv_rate": g["is_sv"].mean(),
        "approach_rate": g["has_approach"].mean(),
        "short_point_rate": short_pts / n,
        "long_rally_rate": long_pts / n,
        "avg_rally_shots": rs.mean(),
        "net_finish_rate": last_net / n,
        "slice_finish_rate": last_slice / n,
        "drop_finish_rate": last_drop / n,
        "ground_finish_rate": last_ground / n,
    })

parts = []

usecols = [
    "match_id","Year","Surface","Svr","Player 1","Player 2",
    "rally_shots","is_sv","has_approach","last_shot_type","point_end"
]

for i, chunk in enumerate(pd.read_csv(INPUT_PATH, usecols=usecols, chunksize=CHUNKSIZE), start=1):
    print("Chunk", i, "rows:", len(chunk))

    chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
    chunk = chunk.dropna(subset=["Year","Svr","Player 1","Player 2"])
    chunk["Year"] = chunk["Year"].astype(int)

    chunk["Svr"] = pd.to_numeric(chunk["Svr"], errors="coerce")
    chunk = chunk.dropna(subset=["Svr"])
    chunk["Svr"] = chunk["Svr"].astype(int)

    chunk["is_sv"] = chunk["is_sv"].astype(str).str.lower().isin(["true","1"])
    chunk["has_approach"] = chunk["has_approach"].astype(str).str.lower().isin(["true","1"])

    pr = role_split(chunk)

    agg = (
        pr.groupby(["player","Year","role"], as_index=False)
          .apply(summarize_style)
          .reset_index(drop=True)
    )

    parts.append(agg)

all_parts = pd.concat(parts, ignore_index=True)

group_cols = ["player","Year","role"]

def weighted_mean(col):
    return np.average(all_parts[col], weights=all_parts["points"])

final = (
    all_parts.groupby(group_cols, as_index=False)
             .apply(lambda g: pd.Series({
                 "points": g["points"].sum(),
                 "serve_only_rate": np.average(g["serve_only_rate"], weights=g["points"]),
                 "sv_rate": np.average(g["sv_rate"], weights=g["points"]),
                 "approach_rate": np.average(g["approach_rate"], weights=g["points"]),
                 "short_point_rate": np.average(g["short_point_rate"], weights=g["points"]),
                 "long_rally_rate": np.average(g["long_rally_rate"], weights=g["points"]),
                 "avg_rally_shots": np.average(g["avg_rally_shots"], weights=g["points"]),
                 "net_finish_rate": np.average(g["net_finish_rate"], weights=g["points"]),
                 "slice_finish_rate": np.average(g["slice_finish_rate"], weights=g["points"]),
                 "drop_finish_rate": np.average(g["drop_finish_rate"], weights=g["points"]),
                 "ground_finish_rate": np.average(g["ground_finish_rate"], weights=g["points"]),
             }))
             .reset_index(drop=True)
)

final = final[final["points"] >= MIN_POINTS_PLAYER_YEAR].copy()

final.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH, "rows:", len(final))