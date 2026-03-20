"""
NCAA March Madness 2026 - Data Loading & Preprocessing
Loads all CSV files, unifies M/W data, builds symmetric training pairs.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "march-machine-learning-mania-2026"


def load_all_data():
    """Load all CSV files into a dictionary of DataFrames."""
    data = {}
    for f in DATA_DIR.glob("*.csv"):
        key = f.stem
        data[key] = pd.read_csv(f, encoding="utf-8")
    print(f"Loaded {len(data)} files: {sorted(data.keys())}")
    return data


def combine_gender(m_df, w_df, add_gender_col=True):
    """Combine men's and women's DataFrames. They share same schema."""
    if add_gender_col:
        m_df = m_df.copy()
        w_df = w_df.copy()
        m_df["Gender"] = "M"
        w_df["Gender"] = "W"
    return pd.concat([m_df, w_df], ignore_index=True)


def load_regular_season_detailed(data):
    """Load and combine M+W regular season detailed results."""
    m = data["MRegularSeasonDetailedResults"].copy()
    w = data["WRegularSeasonDetailedResults"].copy()
    return combine_gender(m, w)


def load_regular_season_compact(data):
    """Load and combine M+W regular season compact results."""
    m = data["MRegularSeasonCompactResults"].copy()
    w = data["WRegularSeasonCompactResults"].copy()
    return combine_gender(m, w)


def load_tourney_compact(data):
    """Load and combine M+W NCAA tournament compact results."""
    m = data["MNCAATourneyCompactResults"].copy()
    w = data["WNCAATourneyCompactResults"].copy()
    return combine_gender(m, w)


def load_tourney_detailed(data):
    """Load and combine M+W NCAA tournament detailed results."""
    m = data["MNCAATourneyDetailedResults"].copy()
    w = data["WNCAATourneyDetailedResults"].copy()
    return combine_gender(m, w)


def load_seeds(data):
    """Load and combine M+W tournament seeds, parse seed number."""
    m = data["MNCAATourneySeeds"].copy()
    w = data["WNCAATourneySeeds"].copy()
    seeds = combine_gender(m, w)
    # Parse seed: "W01a" -> region='W', seed_num=1
    seeds["Region"] = seeds["Seed"].str[0]
    seeds["SeedNum"] = seeds["Seed"].str[1:3].astype(int)
    seeds["PlayIn"] = seeds["Seed"].str[3:].apply(lambda x: x if x else "")
    return seeds


def load_massey_ordinals(data):
    """Load Massey Ordinals (men only, ~5.8M rows)."""
    return data["MMasseyOrdinals"].copy()


def load_coaches(data):
    """Load men's coaching data."""
    return data["MTeamCoaches"].copy()


def load_conferences(data):
    """Load team conference affiliations (M+W)."""
    m = data["MTeamConferences"].copy()
    w = data["WTeamConferences"].copy()
    return combine_gender(m, w)


def load_conference_tourney(data):
    """Load conference tournament games (M+W)."""
    m = data["MConferenceTourneyGames"].copy()
    w = data["WConferenceTourneyGames"].copy()
    return combine_gender(m, w)


def load_teams(data):
    """Load team info (M+W)."""
    m = data["MTeams"].copy()
    w = data["WTeams"].copy()
    m["Gender"] = "M"
    w["Gender"] = "W"
    return m, w


def build_tourney_labels(tourney_compact):
    """
    Build tournament matchup labels in submission format.
    ID: SSSS_XXXX_YYYY where XXXX < YYYY
    Label: 1 if lower TeamID won, 0 otherwise
    """
    df = tourney_compact.copy()
    rows = []
    for _, r in df.iterrows():
        season = r["Season"]
        low_id = min(r["WTeamID"], r["LTeamID"])
        high_id = max(r["WTeamID"], r["LTeamID"])
        label = 1 if r["WTeamID"] == low_id else 0
        game_id = f"{season}_{low_id}_{high_id}"
        rows.append({
            "ID": game_id,
            "Season": season,
            "TeamA": low_id,   # lower ID
            "TeamB": high_id,  # higher ID
            "Label": label,    # P(TeamA wins)
            "Gender": r.get("Gender", "M"),
            # ScoreDiff from TeamA (low ID) perspective: positive = TeamA won
            "ScoreDiff": (r["WScore"] - r["LScore"]) if r["WTeamID"] == low_id else (r["LScore"] - r["WScore"]),
        })
    return pd.DataFrame(rows)


def parse_submission(filepath):
    """Parse sample submission file to get required matchup IDs."""
    df = pd.read_csv(filepath)
    parts = df["ID"].str.split("_", expand=True)
    df["Season"] = parts[0].astype(int)
    df["TeamA"] = parts[1].astype(int)
    df["TeamB"] = parts[2].astype(int)
    return df


if __name__ == "__main__":
    data = load_all_data()
    rs = load_regular_season_detailed(data)
    print(f"\nRegular Season Detailed: {len(rs)} games")
    print(f"  Men: {(rs.Gender=='M').sum()}, Women: {(rs.Gender=='W').sum()}")
    print(f"  Seasons: {rs.Season.min()}-{rs.Season.max()}")

    tc = load_tourney_compact(data)
    labels = build_tourney_labels(tc)
    print(f"\nTourney Labels: {len(labels)} games")
    print(f"  Label distribution: {labels.Label.mean():.3f} (lower ID win rate)")

    seeds = load_seeds(data)
    print(f"\nSeeds: {len(seeds)} entries")
    print(f"  Seed range: {seeds.SeedNum.min()}-{seeds.SeedNum.max()}")
