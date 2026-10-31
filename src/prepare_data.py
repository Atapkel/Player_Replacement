"""Shrink the 5.6 GB male_players.csv into a small parquet file.

The raw file stores every weekly ratings update EA made during each FIFA version
(about 10 million rows). For the project we only need two snapshots per player per
version:

    start  - earliest update of the version (around game launch, September)
    end    - latest update of the version (end of season, close to the summer
             transfer window)

The file is read in chunks so it never has to fit in memory at once.

Output: data/processed/players.parquet, one row per (player_id, fifa_version, snapshot).
"""

import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "male_players.csv"
OUT_FILE = ROOT / "data" / "processed" / "players.parquet"

CHUNK_ROWS = 500_000
KEY = ["player_id", "fifa_version"]

ID_COLS = [
    "player_id", "fifa_version", "fifa_update", "fifa_update_date",
    "short_name", "long_name", "dob", "nationality_name",
]
CLUB_COLS = [
    "league_id", "league_name", "league_level", "club_team_id", "club_name",
    "club_position", "club_loaned_from", "club_joined_date",
    "club_contract_valid_until_year",
]
PROFILE_COLS = [
    "player_positions", "overall", "potential", "value_eur", "wage_eur",
    "release_clause_eur", "age", "height_cm", "weight_kg", "preferred_foot",
    "weak_foot", "skill_moves", "international_reputation", "work_rate",
    "player_traits",
]
MAIN_STATS = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
DETAILED_SKILLS = [
    "attacking_crossing", "attacking_finishing", "attacking_heading_accuracy",
    "attacking_short_passing", "attacking_volleys",
    "skill_dribbling", "skill_curve", "skill_fk_accuracy", "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration", "movement_sprint_speed", "movement_agility",
    "movement_reactions", "movement_balance",
    "power_shot_power", "power_jumping", "power_stamina", "power_strength",
    "power_long_shots",
    "mentality_aggression", "mentality_interceptions", "mentality_positioning",
    "mentality_vision", "mentality_penalties", "mentality_composure",
    "defending_marking_awareness", "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving", "goalkeeping_handling", "goalkeeping_kicking",
    "goalkeeping_positioning", "goalkeeping_reflexes", "goalkeeping_speed",
]
USE_COLS = ID_COLS + CLUB_COLS + PROFILE_COLS + MAIN_STATS + DETAILED_SKILLS

# Ratings are 0-100 and some are missing (e.g. goalkeepers have no "pace"),
# so float32 is enough and halves memory compared with the default float64.
RATING_COLS = MAIN_STATS + DETAILED_SKILLS + [
    "overall", "potential", "age", "height_cm", "weight_kg", "weak_foot",
    "skill_moves", "international_reputation", "league_level",
    "club_contract_valid_until_year",
]
DTYPES = {c: "float32" for c in RATING_COLS} | {
    "player_id": "int32", "fifa_version": "int8", "fifa_update": "int16",
}

POSITION_GROUP = {
    "GK": "GK",
    "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
    "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
    "LW": "FWD", "RW": "FWD", "CF": "FWD", "ST": "FWD",
}


def pick_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest and the latest update of every (player, version)."""
    groups = df.groupby(KEY)["fifa_update"]
    start = df.loc[groups.idxmin()].assign(snapshot="start")
    end = df.loc[groups.idxmax()].assign(snapshot="end")
    return pd.concat([start, end], ignore_index=True)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    # FIFA 20 covers the 2019/20 season, so the season starts in 2019
    df["season_start"] = (2000 + df["fifa_version"].astype(int) - 1).astype("int16")
    df["main_position"] = df["player_positions"].str.split(",").str[0].str.strip()
    df["position_group"] = df["main_position"].map(POSITION_GROUP)
    df["contract_years_left"] = df["club_contract_valid_until_year"] - df["season_start"]
    return df


def main():
    t0 = time.time()
    reader = pd.read_csv(RAW_FILE, usecols=USE_COLS, dtype=DTYPES, chunksize=CHUNK_ROWS)

    # A player's updates can be split across two chunks, so reduce each chunk
    # first and then reduce the combined candidates once more at the end.
    candidates = []
    rows_read = 0
    for i, chunk in enumerate(reader):
        rows_read += len(chunk)
        candidates.append(pick_snapshots(chunk))
        print(f"chunk {i:3d}: {rows_read:>11,} rows read, {time.time() - t0:5.0f} s")

    combined = pd.concat(candidates, ignore_index=True)
    start = combined[combined.snapshot == "start"]
    end = combined[combined.snapshot == "end"]
    start = start.loc[start.groupby(KEY)["fifa_update"].idxmin()]
    end = end.loc[end.groupby(KEY)["fifa_update"].idxmax()]
    players = pd.concat([start, end], ignore_index=True)

    players = add_basic_features(players)
    text_cols = players.select_dtypes("object").columns.drop(
        ["short_name", "long_name", "player_traits"]
    )
    players[text_cols] = players[text_cols].astype("category")
    players = players.sort_values(["fifa_version", "snapshot", "player_id"]).reset_index(drop=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    players.to_parquet(OUT_FILE, index=False)

    print(f"\nRaw rows read:  {rows_read:,}")
    print(f"Rows kept:      {len(players):,}  ({players.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory)")
    print(f"Saved to:       {OUT_FILE}  ({OUT_FILE.stat().st_size / 1e6:.0f} MB on disk)")
    print(f"Total time:     {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
