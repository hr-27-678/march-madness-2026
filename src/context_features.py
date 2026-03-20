"""
NCAA March Madness 2026 - Contextual Features
Coach experience, conference strength, seed features.
"""

import pandas as pd
import numpy as np


def compute_seed_features(seeds_df):
    """
    Extract seed-related features.

    Returns: DataFrame (Season, TeamID) -> seed features
    """
    df = seeds_df[["Season", "TeamID", "SeedNum", "Region"]].copy()
    df = df.rename(columns={"SeedNum": "seed"})

    # Region as category (W=0, X=1, Y=2, Z=3)
    region_map = {"W": 0, "X": 1, "Y": 2, "Z": 3}
    df["region_code"] = df["Region"].map(region_map)

    # Seed quality bucket
    df["seed_top4"] = (df["seed"] <= 4).astype(int)
    df["seed_top8"] = (df["seed"] <= 8).astype(int)

    return df[["Season", "TeamID", "seed", "region_code", "seed_top4", "seed_top8"]]


def compute_coach_features(coaches_df):
    """
    Compute coaching features from MTeamCoaches.csv.

    Features:
    - coach_seasons: total seasons coached (historical)
    - coach_tenure: years at current team
    - is_new_coach: first season at this team?
    """
    df = coaches_df.copy()

    # Only keep the "main" coach (covers DayNum 0 to 154 or close)
    # Filter to the coach with the longest tenure in the season
    df["duration"] = df["LastDayNum"] - df["FirstDayNum"]
    idx = df.groupby(["Season", "TeamID"])["duration"].idxmax()
    main_coaches = df.loc[idx, ["Season", "TeamID", "CoachName"]].copy()

    # Coach total seasons (across all teams)
    coach_total = main_coaches.groupby("CoachName").size().reset_index(name="coach_total_seasons")
    # But we need cumulative up to each season
    main_coaches = main_coaches.sort_values(["CoachName", "Season"])
    main_coaches["coach_career_year"] = main_coaches.groupby("CoachName").cumcount() + 1

    # Coach tenure at current team
    # Group consecutive seasons at same team
    main_coaches = main_coaches.sort_values(["TeamID", "CoachName", "Season"])

    def calc_tenure(group):
        group = group.sort_values("Season")
        tenure = []
        count = 0
        for _, row in group.iterrows():
            count += 1
            tenure.append(count)
        group["coach_tenure"] = tenure
        return group

    main_coaches = main_coaches.groupby(["TeamID", "CoachName"], group_keys=False).apply(calc_tenure)

    # Is new coach?
    main_coaches["is_new_coach"] = (main_coaches["coach_tenure"] == 1).astype(int)

    # Coach tournament history: count past tournament appearances
    # (We'll compute this separately when we have tournament data)

    result = main_coaches[["Season", "TeamID", "CoachName", "coach_career_year",
                           "coach_tenure", "is_new_coach"]].copy()

    return result


def compute_conference_strength(team_conferences, basic_stats):
    """
    Compute conference-level strength features.

    Features:
    - conf_avg_win_rate: average win rate of teams in the conference
    - conf_avg_margin: average scoring margin of conference teams
    - conf_num_teams: number of teams in the conference
    """
    # Merge conference info with basic stats
    merged = team_conferences.merge(
        basic_stats[["Season", "TeamID", "win_rate", "avg_margin"]],
        on=["Season", "TeamID"],
        how="left"
    )

    # Aggregate by conference
    conf_strength = merged.groupby(["Season", "ConfAbbrev"]).agg(
        conf_avg_win_rate=("win_rate", "mean"),
        conf_median_win_rate=("win_rate", "median"),
        conf_avg_margin=("avg_margin", "mean"),
        conf_top_win_rate=("win_rate", "max"),
        conf_num_teams=("TeamID", "count"),
    ).reset_index()

    # Rank conferences by strength
    conf_strength["conf_rank"] = conf_strength.groupby("Season")["conf_avg_win_rate"].rank(
        ascending=False, method="min"
    )

    # Merge back to team level
    result = team_conferences[["Season", "TeamID", "ConfAbbrev"]].merge(
        conf_strength, on=["Season", "ConfAbbrev"], how="left"
    )

    return result


def compute_coach_tourney_history(coaches, tourney_compact):
    """
    Count how many times each coach has been in the NCAA tournament historically.
    """
    # Get coach for each team-season
    df = coaches.copy()
    df["duration"] = df["LastDayNum"] - df["FirstDayNum"]
    idx = df.groupby(["Season", "TeamID"])["duration"].idxmax()
    main_coaches = df.loc[idx, ["Season", "TeamID", "CoachName"]].copy()

    # Get teams that made the tournament each year (from tourney results)
    tourney_teams_w = tourney_compact[["Season", "WTeamID"]].rename(columns={"WTeamID": "TeamID"})
    tourney_teams_l = tourney_compact[["Season", "LTeamID"]].rename(columns={"LTeamID": "TeamID"})
    tourney_teams = pd.concat([tourney_teams_w, tourney_teams_l]).drop_duplicates()
    tourney_teams["in_tourney"] = 1

    # Merge
    coach_tourney = main_coaches.merge(tourney_teams, on=["Season", "TeamID"], how="left")
    coach_tourney["in_tourney"] = coach_tourney["in_tourney"].fillna(0)

    # Cumulative tournament appearances by coach
    coach_tourney = coach_tourney.sort_values(["CoachName", "Season"])
    coach_tourney["coach_tourney_apps"] = coach_tourney.groupby("CoachName")["in_tourney"].cumsum()
    # Shift by 1 so we don't leak current season
    coach_tourney["coach_tourney_apps"] = coach_tourney.groupby("CoachName")["coach_tourney_apps"].shift(1).fillna(0)

    return coach_tourney[["Season", "TeamID", "coach_tourney_apps"]]
