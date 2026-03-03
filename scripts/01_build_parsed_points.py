import pandas as pd
import numpy as np

# --------------------
# Paths
# --------------------
POINTS_FILES = [
    "data/raw/charting-m-points-to-2009.csv",
    "data/raw/charting-m-points-2010s.csv",
    "data/raw/charting-m-points-2020s.csv",
]
MATCHES_PATH = "data/raw/charting-m-matches.csv"

# Write a single big processed file (CSV). Later we can convert to parquet.
OUT_PATH = "data/processed/points_all_parsed.csv"

# --------------------
# Config
# --------------------
CHUNKSIZE = 250000  # adjust up/down depending on your RAM

VALID_SURFACES = set(["Hard", "Clay", "Grass"])
SERVE_DIR = set(["0", "4", "5", "6"])
SERVE_FAULT = set(["n", "w", "d", "x", "g", "e", "!"])
RALLY_SHOTS = set(list("fbrsvzopuy lmhiqjkt".replace(" ", "")) + ["q"])
END_MARKERS = set(["*", "@", "#"])


def as_clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def pick_point_string(row):
    s2 = as_clean_str(row["2nd"])
    s1 = as_clean_str(row["1st"])
    if s2 != "":
        return s2
    return s1


def parse_point(point_string):
    s = as_clean_str(point_string)

    serve_dir = np.nan
    is_sv = False
    serve_fault_type = ""
    is_ace = False
    is_unreturned = False
    rally_shots = 0
    point_end = "unknown"
    last_shot_type = ""
    has_approach = False

    if s == "":
        return (serve_dir, is_sv, serve_fault_type, is_ace, is_unreturned,
                rally_shots, point_end, last_shot_type, has_approach)

    i = 0

    # let serves
    while i < len(s) and s[i] == "c":
        i += 1

    # serve direction
    if i < len(s) and s[i] in SERVE_DIR:
        serve_dir = int(s[i])
        i += 1
    else:
        return (serve_dir, is_sv, serve_fault_type, is_ace, is_unreturned,
                rally_shots, point_end, last_shot_type, has_approach)

    # serve-and-volley
    if i < len(s) and s[i] == "+":
        is_sv = True
        i += 1

    # serve fault-only strings
    if i < len(s) and s[i] in SERVE_FAULT:
        serve_fault_type = s[i]
        point_end = "serve_only"
        return (serve_dir, is_sv, serve_fault_type, is_ace, is_unreturned,
                rally_shots, point_end, last_shot_type, has_approach)

    # ace/unreturned
    if i < len(s) and s[i] == "*":
        is_ace = True
        point_end = "serve_only"
        return (serve_dir, is_sv, serve_fault_type, is_ace, is_unreturned,
                rally_shots, point_end, last_shot_type, has_approach)

    if i < len(s) and s[i] == "#":
        is_unreturned = True
        point_end = "serve_only"
        return (serve_dir, is_sv, serve_fault_type, is_ace, is_unreturned,
                rally_shots, point_end, last_shot_type, has_approach)

    end_marker = ""

    while i < len(s):
        ch = s[i]

        if ch in END_MARKERS:
            end_marker = ch
            i += 1
            continue

        if ch in RALLY_SHOTS:
            shot = ch
            i += 1

            # modifiers
            while i < len(s) and s[i] in set(["+", "-", "=", ";", "^"]):
                if s[i] == "+":
                    has_approach = True
                i += 1

            # optional direction 0-3
            if i < len(s) and s[i] in set(["0", "1", "2", "3"]):
                i += 1
                # optional return depth 7-9
                if i < len(s) and s[i] in set(["7", "8", "9"]):
                    i += 1

            # optional error type
            if i < len(s) and s[i] in SERVE_FAULT:
                i += 1

            rally_shots += 1
            last_shot_type = shot
            continue

        i += 1

    if end_marker == "*":
        point_end = "winner"
    elif end_marker == "@":
        point_end = "unforced_error"
    elif end_marker == "#":
        point_end = "forced_error"
    else:
        point_end = "unknown"

    return (serve_dir, is_sv, serve_fault_type, is_ace, is_unreturned,
            rally_shots, point_end, last_shot_type, has_approach)


def load_and_clean_matches(matches_path):
    matches = pd.read_csv(matches_path, low_memory=False)

    for col in ["match_id", "Surface", "Date", "Player 1", "Player 2", "Tournament", "Round"]:
        if col in matches.columns:
            matches[col] = matches[col].astype(str).str.strip()

    matches = matches[matches["Surface"].isin(VALID_SURFACES)].copy()

    matches["Year"] = matches["Date"].str.slice(0, 4)
    matches = matches[matches["Year"].str.fullmatch(r"\d{4}", na=False)].copy()
    matches["Year"] = matches["Year"].astype(int)

    keep_cols = ["match_id", "Player 1", "Player 2", "Surface", "Year", "Tournament", "Round", "Date"]
    keep_cols = [c for c in keep_cols if c in matches.columns]
    return matches[keep_cols].copy()


def parse_chunk(df):
    # clean strings
    df["1st"] = df["1st"].apply(as_clean_str)
    df["2nd"] = df["2nd"].apply(as_clean_str)

    # build point_string
    df["point_string"] = df.apply(pick_point_string, axis=1)

    # parse (returns tuples)
    parsed = df["point_string"].map(parse_point)

    # split tuples into columns (fast)
    cols = [
        "serve_dir", "is_sv", "serve_fault_type", "is_ace", "is_unreturned",
        "rally_shots", "point_end", "last_shot_type", "has_approach"
    ]
    parsed_df = pd.DataFrame(parsed.tolist(), columns=cols, index=df.index)

    return pd.concat([df, parsed_df], axis=1)


# --------------------
# Main: chunked pipeline
# --------------------
matches = load_and_clean_matches(MATCHES_PATH)

usecols = ["match_id", "Pt", "Set1", "Set2", "Gm1", "Gm2", "Pts", "Gm#", "TbSet", "Svr", "1st", "2nd", "Notes", "PtWinner"]

# overwrite output each run
first_write = True

for fp in POINTS_FILES:
    print("\nReading:", fp)

    chunk_iter = pd.read_csv(
        fp,
        usecols=usecols,
        dtype={
            "match_id": "string",
            "1st": "string",
            "2nd": "string",
            "Notes": "string",
            "Pts": "string",
            "TbSet": "string",
        },
        low_memory=False,
        chunksize=CHUNKSIZE
    )

    for chunk_i, chunk in enumerate(chunk_iter, start=1):
        print("  Chunk", chunk_i, "rows:", len(chunk))

        chunk["match_id"] = chunk["match_id"].astype(str).str.strip()
        chunk["source_file"] = fp

        # numeric coercions (optional)
        for col in ["Pt", "Set1", "Set2", "Gm1", "Gm2", "Gm#", "Svr", "PtWinner"]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

        # parse
        parsed_chunk = parse_chunk(chunk)

        # merge match metadata
        parsed_chunk = parsed_chunk.merge(matches, on="match_id", how="left")
        parsed_chunk = parsed_chunk[~parsed_chunk["Surface"].isna()].copy()

        # write incrementally
        mode = "w" if first_write else "a"
        header = True if first_write else False

        parsed_chunk.to_csv(OUT_PATH, index=False, mode=mode, header=header)
        first_write = False

print("\nDone. Wrote:", OUT_PATH)

# Optional quick sanity check (reads only a small sample from the output)
sample = pd.read_csv(OUT_PATH, nrows=200000)
print("Sample rows:", len(sample))
print("Sample matches:", sample["match_id"].nunique())
print("Sample surface counts:")
print(sample["Surface"].value_counts())
print("Sample point_end counts:")
print(sample["point_end"].value_counts().head(10))