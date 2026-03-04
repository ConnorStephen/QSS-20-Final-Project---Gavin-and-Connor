import pandas as pd
import numpy as np


INPUT_PATH = "data/processed/points_all_parsed.csv"   
OUTPUT_PATH = "outputs/player_surface_year.csv"       

# Process in chunks so it doesn't blow RAM
CHUNKSIZE = 400000  

# helpers
def safe_div(numer, denom):
    if denom == 0 or pd.isna(denom):
        return np.nan
    return numer / denom


def add_player_rows(df, player_col, is_server_side):
    """
    Convert match-level Player 1 / Player 2 into player-level rows.

    is_server_side=True  -> keep points where this player served (Svr indicates server: 1 or 2)
    is_server_side=False -> keep points where this player returned
    """
    out = df.copy()
    out["player"] = out[player_col]

    # Svr == 1 means Player 1 served; Svr == 2 means Player 2 served
    if is_server_side:
        if player_col == "Player 1":
            out = out[out["Svr"] == 1]
        else:
            out = out[out["Svr"] == 2]
        out["role"] = "server"
    else:
        if player_col == "Player 1":
            out = out[out["Svr"] == 2]
        else:
            out = out[out["Svr"] == 1]
        out["role"] = "returner"

    return out


def summarize_group(g):
    """
    Compute aggregated metrics for a player-year-surface group.
    """
    n = len(g)

    # Serve direction distribution 
    serve_dir_valid = g["serve_dir"].dropna()
    n_dir = len(serve_dir_valid)

    pct_serve_4 = safe_div((serve_dir_valid == 4).sum(), n_dir)
    pct_serve_5 = safe_div((serve_dir_valid == 5).sum(), n_dir)
    pct_serve_6 = safe_div((serve_dir_valid == 6).sum(), n_dir)
    pct_serve_0 = safe_div((serve_dir_valid == 0).sum(), n_dir)

    # End types
    serve_only = (g["point_end"] == "serve_only").sum()
    winners = (g["point_end"] == "winner").sum()
    ues = (g["point_end"] == "unforced_error").sum()
    fes = (g["point_end"] == "forced_error").sum()
    unknown = (g["point_end"] == "unknown").sum()

    # Serve outcomes
    aces = g["is_ace"].sum()
    unreturned = g["is_unreturned"].sum()

    # Serve-and-volley + approach
    sv = g["is_sv"].sum()
    approaches = g["has_approach"].sum()

    # Rally length
    avg_rally = g["rally_shots"].mean()

    return pd.Series({
        "points": n,
        "ace_rate": safe_div(aces, n),
        "unreturned_rate": safe_div(unreturned, n),
        "serve_only_rate": safe_div(serve_only, n),
        "winner_rate": safe_div(winners, n),
        "ue_rate": safe_div(ues, n),
        "fe_rate": safe_div(fes, n),
        "unknown_rate": safe_div(unknown, n),
        "sv_rate": safe_div(sv, n),
        "approach_rate": safe_div(approaches, n),
        "avg_rally_shots": avg_rally,
        "pct_serve_4": pct_serve_4,
        "pct_serve_5": pct_serve_5,
        "pct_serve_6": pct_serve_6,
        "pct_serve_0": pct_serve_0,
    })


def reaggregate(df):
    """
    Re-aggregate chunk-level aggregates into final aggregates.
    We recompute:
      - points: sum
      - rates/avgs: weighted average by points
    """
    group_cols = ["player", "Year", "Surface", "role"]

    rate_cols = [
        "ace_rate", "unreturned_rate", "serve_only_rate",
        "winner_rate", "ue_rate", "fe_rate", "unknown_rate",
        "sv_rate", "approach_rate",
        "pct_serve_4", "pct_serve_5", "pct_serve_6", "pct_serve_0"
    ]
    avg_cols = ["avg_rally_shots"]

    out = (
        df.groupby(group_cols, as_index=False)["points"]
        .sum()
    )

    def weighted_col(colname):
        s = (
            df.groupby(group_cols, group_keys=False)
              .apply(lambda g: np.average(g[colname], weights=g["points"]) if g["points"].sum() > 0 else np.nan)
        )
        return s.rename(colname).reset_index()

    for c in rate_cols + avg_cols:
        out = out.merge(weighted_col(c), on=group_cols, how="left")

    return out


# chunked aggregation
server_aggs = []
returner_aggs = []

usecols = [
    "match_id", "Svr", "Player 1", "Player 2", "Surface", "Year",
    "serve_dir", "is_sv", "serve_fault_type", "is_ace", "is_unreturned",
    "rally_shots", "point_end", "last_shot_type", "has_approach"
]

reader = pd.read_csv(INPUT_PATH, usecols=usecols, chunksize=CHUNKSIZE)

for i, chunk in enumerate(reader, start=1):
    print("Chunk", i, "rows:", len(chunk))

    chunk["Svr"] = pd.to_numeric(chunk["Svr"], errors="coerce")
    chunk["Year"] = pd.to_numeric(chunk["Year"], errors="coerce")
    chunk["serve_dir"] = pd.to_numeric(chunk["serve_dir"], errors="coerce")

    chunk["is_sv"] = chunk["is_sv"].astype(str).str.lower().isin(["true", "1"])
    chunk["is_ace"] = chunk["is_ace"].astype(str).str.lower().isin(["true", "1"])
    chunk["is_unreturned"] = chunk["is_unreturned"].astype(str).str.lower().isin(["true", "1"])
    chunk["has_approach"] = chunk["has_approach"].astype(str).str.lower().isin(["true", "1"])

    chunk["rally_shots"] = pd.to_numeric(chunk["rally_shots"], errors="coerce")

    chunk = chunk.dropna(subset=["Svr", "Year", "Surface", "Player 1", "Player 2"])

    # Build server-side rows for Player 1 and Player 2
    s1 = add_player_rows(chunk, "Player 1", is_server_side=True)
    s2 = add_player_rows(chunk, "Player 2", is_server_side=True)
    servers = pd.concat([s1, s2], ignore_index=True)

    # Build returner-side rows for Player 1 and Player 2
    r1 = add_player_rows(chunk, "Player 1", is_server_side=False)
    r2 = add_player_rows(chunk, "Player 2", is_server_side=False)
    returners = pd.concat([r1, r2], ignore_index=True)

    # Aggregate within chunk
    server_chunk_agg = (
        servers
        .groupby(["player", "Year", "Surface"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    server_chunk_agg["role"] = "server"
    server_aggs.append(server_chunk_agg)

    returner_chunk_agg = (
        returners
        .groupby(["player", "Year", "Surface"], as_index=False)
        .apply(summarize_group)
        .reset_index(drop=True)
    )
    returner_chunk_agg["role"] = "returner"
    returner_aggs.append(returner_chunk_agg)

server_all = pd.concat(server_aggs, ignore_index=True)
returner_all = pd.concat(returner_aggs, ignore_index=True)

server_final = reaggregate(server_all)
returner_final = reaggregate(returner_all)

final = pd.concat([server_final, returner_final], ignore_index=True)

final.to_csv(OUTPUT_PATH, index=False)
print("Wrote:", OUTPUT_PATH)
print("Rows:", len(final))
print(final.head())