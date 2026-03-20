"""
NCAA March Madness 2026 - Feature Engineering
All feature categories: basic stats, advanced stats, Four Factors, efficiency.
Computes team-season level features from regular season detailed results.
"""

import pandas as pd
import numpy as np
from functools import reduce


# ============================================================================
# 1. BASIC SEASON STATISTICS
# ============================================================================

def compute_basic_stats(detailed_results, compact_results=None):
    """
    Compute basic season-level stats for each team.
    Uses detailed results primarily, compact for win/loss not in detailed.

    Returns: DataFrame with (Season, TeamID) as index
    """
    df = detailed_results.copy()

    # --- Build team-game rows (one row per team per game) ---
    # Winner perspective
    w_cols = {
        "Season": "Season", "DayNum": "DayNum", "WTeamID": "TeamID",
        "WScore": "Score", "LScore": "OppScore", "LTeamID": "OppTeamID",
        "WLoc": "Loc", "NumOT": "NumOT", "Gender": "Gender",
        "WFGM": "FGM", "WFGA": "FGA", "WFGM3": "FGM3", "WFGA3": "FGA3",
        "WFTM": "FTM", "WFTA": "FTA", "WOR": "OR", "WDR": "DR",
        "WAst": "Ast", "WTO": "TO", "WStl": "Stl", "WBlk": "Blk", "WPF": "PF",
        "LFGM": "OppFGM", "LFGA": "OppFGA", "LFGM3": "OppFGM3", "LFGA3": "OppFGA3",
        "LFTM": "OppFTM", "LFTA": "OppFTA", "LOR": "OppOR", "LDR": "OppDR",
        "LAst": "OppAst", "LTO": "OppTO", "LStl": "OppStl", "LBlk": "OppBlk", "LPF": "OppPF",
    }
    winners = df.rename(columns=w_cols)[[v for v in w_cols.values()]].copy()
    winners["Win"] = 1
    winners["Loc_H"] = (winners["Loc"] == "H").astype(int)
    winners["Loc_A"] = (winners["Loc"] == "A").astype(int)
    winners["Loc_N"] = (winners["Loc"] == "N").astype(int)

    # Loser perspective
    l_cols = {
        "Season": "Season", "DayNum": "DayNum", "LTeamID": "TeamID",
        "LScore": "Score", "WScore": "OppScore", "WTeamID": "OppTeamID",
        "WLoc": "WinnerLoc", "NumOT": "NumOT", "Gender": "Gender",
        "LFGM": "FGM", "LFGA": "FGA", "LFGM3": "FGM3", "LFGA3": "FGA3",
        "LFTM": "FTM", "LFTA": "FTA", "LOR": "OR", "LDR": "DR",
        "LAst": "Ast", "LTO": "TO", "LStl": "Stl", "LBlk": "Blk", "LPF": "PF",
        "WFGM": "OppFGM", "WFGA": "OppFGA", "WFGM3": "OppFGM3", "WFGA3": "OppFGA3",
        "WFTM": "OppFTM", "WFTA": "OppFTA", "WOR": "OppOR", "WDR": "OppDR",
        "WAst": "OppAst", "WTO": "OppTO", "WStl": "OppStl", "WBlk": "OppBlk", "WPF": "OppPF",
    }
    losers = df.rename(columns=l_cols)[[v for v in l_cols.values() if v != "WinnerLoc"]].copy()
    # Fix location for losers: if winner was H, loser was A, etc.
    losers["Win"] = 0
    winner_loc = df["WLoc"].values
    losers["Loc_H"] = (winner_loc == "A").astype(int)
    losers["Loc_A"] = (winner_loc == "H").astype(int)
    losers["Loc_N"] = (winner_loc == "N").astype(int)
    losers["Loc"] = np.where(winner_loc == "H", "A", np.where(winner_loc == "A", "H", "N"))

    # Combine
    team_games = pd.concat([winners, losers], ignore_index=True)
    team_games["Margin"] = team_games["Score"] - team_games["OppScore"]

    return team_games


def aggregate_season_stats(team_games, day_cutoff=133):
    """
    Aggregate team_games into season-level features.
    Only use games up to day_cutoff (default 133 = end of regular season).

    Returns DataFrame indexed by (Season, TeamID).
    """
    df = team_games[team_games["DayNum"] <= day_cutoff].copy()

    # --- Basic aggregations ---
    agg = df.groupby(["Season", "TeamID"]).agg(
        games_played=("Win", "count"),
        wins=("Win", "sum"),
        avg_score=("Score", "mean"),
        avg_opp_score=("OppScore", "mean"),
        avg_margin=("Margin", "mean"),
        std_margin=("Margin", "std"),
        std_score=("Score", "std"),
        max_margin=("Margin", "max"),
        min_margin=("Margin", "min"),
        avg_ot=("NumOT", "mean"),
    ).reset_index()

    agg["win_rate"] = agg["wins"] / agg["games_played"]
    agg["loss_rate"] = 1 - agg["win_rate"]

    # --- Location-specific win rates ---
    for loc_label, loc_col in [("home", "Loc_H"), ("away", "Loc_A"), ("neutral", "Loc_N")]:
        loc_df = df[df[loc_col] == 1].groupby(["Season", "TeamID"]).agg(
            **{f"{loc_label}_games": ("Win", "count"),
               f"{loc_label}_wins": ("Win", "sum")}
        ).reset_index()
        loc_df[f"{loc_label}_win_rate"] = (
            loc_df[f"{loc_label}_wins"] / loc_df[f"{loc_label}_games"]
        )
        agg = agg.merge(loc_df, on=["Season", "TeamID"], how="left")

    # --- Momentum: last 30 days of regular season (DayNum 103-132) ---
    recent = df[df["DayNum"] >= 103]
    recent_agg = recent.groupby(["Season", "TeamID"]).agg(
        recent_games=("Win", "count"),
        recent_wins=("Win", "sum"),
        recent_avg_margin=("Margin", "mean"),
    ).reset_index()
    recent_agg["recent_win_rate"] = recent_agg["recent_wins"] / recent_agg["recent_games"]
    agg = agg.merge(recent_agg, on=["Season", "TeamID"], how="left")

    # --- Close game performance (margin <= 5) ---
    close = df[df["Margin"].abs() <= 5]
    close_agg = close.groupby(["Season", "TeamID"]).agg(
        close_games=("Win", "count"),
        close_wins=("Win", "sum"),
    ).reset_index()
    close_agg["close_win_rate"] = close_agg["close_wins"] / close_agg["close_games"]
    agg = agg.merge(close_agg, on=["Season", "TeamID"], how="left")

    return agg


# ============================================================================
# 2. ADVANCED BASKETBALL STATISTICS (FOUR FACTORS + EFFICIENCY)
# ============================================================================

def compute_advanced_stats(team_games, day_cutoff=133):
    """
    Compute per-game advanced stats, then aggregate to season level.
    Includes Four Factors, tempo, efficiency ratings.
    """
    df = team_games[team_games["DayNum"] <= day_cutoff].copy()

    # --- Per-game advanced stats ---
    # Possessions estimate (Dean Oliver formula)
    df["Poss"] = df["FGA"] - df["OR"] + df["TO"] + 0.44 * df["FTA"]
    df["OppPoss"] = df["OppFGA"] - df["OppOR"] + df["OppTO"] + 0.44 * df["OppFTA"]
    # Average possessions (both teams should be similar)
    df["AvgPoss"] = (df["Poss"] + df["OppPoss"]) / 2
    df["AvgPoss"] = df["AvgPoss"].clip(lower=1)  # avoid div by zero

    # --- Four Factors (Offense) ---
    df["eFG_pct"] = (df["FGM"] + 0.5 * df["FGM3"]) / df["FGA"].clip(lower=1)
    df["TO_rate"] = df["TO"] / (df["FGA"] + 0.44 * df["FTA"] + df["TO"]).clip(lower=1)
    df["OR_pct"] = df["OR"] / (df["OR"] + df["OppDR"]).clip(lower=1)
    df["FT_rate"] = df["FTM"] / df["FGA"].clip(lower=1)

    # --- Four Factors (Defense - opponent's offense) ---
    df["opp_eFG_pct"] = (df["OppFGM"] + 0.5 * df["OppFGM3"]) / df["OppFGA"].clip(lower=1)
    df["opp_TO_rate"] = df["OppTO"] / (df["OppFGA"] + 0.44 * df["OppFTA"] + df["OppTO"]).clip(lower=1)
    df["opp_OR_pct"] = df["OppOR"] / (df["OppOR"] + df["DR"]).clip(lower=1)
    df["opp_FT_rate"] = df["OppFTM"] / df["OppFGA"].clip(lower=1)

    # --- Efficiency (per 100 possessions) ---
    df["off_efficiency"] = df["Score"] / df["AvgPoss"] * 100
    df["def_efficiency"] = df["OppScore"] / df["AvgPoss"] * 100
    df["net_efficiency"] = df["off_efficiency"] - df["def_efficiency"]

    # --- Tempo ---
    df["tempo"] = df["AvgPoss"]  # raw pace

    # --- Shooting splits ---
    df["fg_pct"] = df["FGM"] / df["FGA"].clip(lower=1)
    df["fg3_pct"] = df["FGM3"] / df["FGA3"].clip(lower=1)
    df["ft_pct"] = df["FTM"] / df["FTA"].clip(lower=1)
    df["fg3_rate"] = df["FGA3"] / df["FGA"].clip(lower=1)  # 3pt dependency
    df["fg2_pct"] = (df["FGM"] - df["FGM3"]) / (df["FGA"] - df["FGA3"]).clip(lower=1)

    # --- Other advanced ---
    df["ast_rate"] = df["Ast"] / df["FGM"].clip(lower=1)
    df["ast_to_ratio"] = df["Ast"] / df["TO"].clip(lower=1)
    df["blk_rate"] = df["Blk"] / df["OppFGA"].clip(lower=1)
    df["stl_rate"] = df["Stl"] / df["OppPoss"].clip(lower=1)
    df["reb_margin"] = (df["OR"] + df["DR"]) - (df["OppOR"] + df["OppDR"])
    df["total_reb"] = df["OR"] + df["DR"]
    df["DR_pct"] = df["DR"] / (df["DR"] + df["OppOR"]).clip(lower=1)

    # --- Opponent shooting splits ---
    df["opp_fg_pct"] = df["OppFGM"] / df["OppFGA"].clip(lower=1)
    df["opp_fg3_pct"] = df["OppFGM3"] / df["OppFGA3"].clip(lower=1)
    df["opp_ft_pct"] = df["OppFTM"] / df["OppFTA"].clip(lower=1)

    # --- Aggregate to season level ---
    adv_cols = [
        "eFG_pct", "TO_rate", "OR_pct", "FT_rate",
        "opp_eFG_pct", "opp_TO_rate", "opp_OR_pct", "opp_FT_rate",
        "off_efficiency", "def_efficiency", "net_efficiency", "tempo",
        "fg_pct", "fg3_pct", "ft_pct", "fg3_rate", "fg2_pct",
        "ast_rate", "ast_to_ratio", "blk_rate", "stl_rate",
        "reb_margin", "total_reb", "DR_pct",
        "opp_fg_pct", "opp_fg3_pct", "opp_ft_pct",
        "Poss",
    ]

    agg_dict = {}
    for col in adv_cols:
        agg_dict[f"{col}_mean"] = (col, "mean")
        agg_dict[f"{col}_std"] = (col, "std")

    adv_agg = df.groupby(["Season", "TeamID"]).agg(**agg_dict).reset_index()

    # --- Recent form (last 30 days) for key metrics ---
    recent = df[df["DayNum"] >= 103]
    key_recent = ["off_efficiency", "def_efficiency", "net_efficiency", "eFG_pct", "TO_rate"]
    recent_dict = {}
    for col in key_recent:
        recent_dict[f"recent_{col}"] = (col, "mean")

    if len(recent) > 0:
        recent_adv = recent.groupby(["Season", "TeamID"]).agg(**recent_dict).reset_index()
        adv_agg = adv_agg.merge(recent_adv, on=["Season", "TeamID"], how="left")

    # --- Trend: compare last 10 games vs first 10 games ---
    def compute_trends(group):
        if len(group) < 10:
            return pd.Series({
                "eff_trend": np.nan,
                "efg_trend": np.nan,
            })
        first_10 = group.head(10)
        last_10 = group.tail(10)
        return pd.Series({
            "eff_trend": last_10["net_efficiency"].mean() - first_10["net_efficiency"].mean(),
            "efg_trend": last_10["eFG_pct"].mean() - first_10["eFG_pct"].mean(),
        })

    sorted_df = df.sort_values(["Season", "TeamID", "DayNum"])
    trends = sorted_df.groupby(["Season", "TeamID"]).apply(
        compute_trends
    ).reset_index()
    adv_agg = adv_agg.merge(trends, on=["Season", "TeamID"], how="left")

    return adv_agg


# ============================================================================
# 3. STRENGTH OF SCHEDULE (SOS)
# ============================================================================

def compute_sos(team_games, basic_stats, day_cutoff=133):
    """
    Compute Strength of Schedule: average win_rate of opponents.
    Also computes SOS-adjusted win rate.
    """
    df = team_games[team_games["DayNum"] <= day_cutoff].copy()

    # Get opponent win rates
    opp_wr = basic_stats[["Season", "TeamID", "win_rate"]].rename(
        columns={"TeamID": "OppTeamID", "win_rate": "opp_win_rate"}
    )

    df = df.merge(opp_wr, on=["Season", "OppTeamID"], how="left")

    sos = df.groupby(["Season", "TeamID"]).agg(
        sos_mean=("opp_win_rate", "mean"),
        sos_median=("opp_win_rate", "median"),
        sos_std=("opp_win_rate", "std"),
        # Wins against strong opponents (>0.6 win rate)
        strong_opp_games=("opp_win_rate", lambda x: (x > 0.6).sum()),
    ).reset_index()

    # Weighted wins: sum of opponent win rates for wins only
    wins_df = df[df["Win"] == 1]
    weighted = wins_df.groupby(["Season", "TeamID"]).agg(
        weighted_wins=("opp_win_rate", "sum"),
        strong_opp_wins=("opp_win_rate", lambda x: (x > 0.6).sum()),
    ).reset_index()

    sos = sos.merge(weighted, on=["Season", "TeamID"], how="left")
    sos["weighted_wins"] = sos["weighted_wins"].fillna(0)
    sos["strong_opp_wins"] = sos["strong_opp_wins"].fillna(0)

    # Strong opp win rate
    sos["strong_opp_win_rate"] = (
        sos["strong_opp_wins"] / sos["strong_opp_games"].clip(lower=1)
    )

    return sos


# ============================================================================
# 4. CONFERENCE TOURNAMENT FEATURES
# ============================================================================

def compute_conf_tourney_features(conf_tourney, conferences):
    """
    Determine if a team won their conference tournament.
    """
    # The last game in each conference tournament = the final
    finals = conf_tourney.sort_values("DayNum").groupby(
        ["Season", "ConfAbbrev"]
    ).last().reset_index()

    # Winner of conference tournament
    conf_champs = finals[["Season", "WTeamID", "Gender"]].rename(
        columns={"WTeamID": "TeamID"}
    )
    conf_champs["conf_tourney_champ"] = 1

    # Also track: did team participate in conf tourney?
    participants_w = conf_tourney[["Season", "WTeamID", "Gender"]].rename(
        columns={"WTeamID": "TeamID"}
    )
    participants_l = conf_tourney[["Season", "LTeamID", "Gender"]].rename(
        columns={"LTeamID": "TeamID"}
    )
    participants = pd.concat([participants_w, participants_l]).drop_duplicates()

    # Count conf tourney wins
    ct_wins = conf_tourney.groupby(["Season", "WTeamID"]).size().reset_index(name="conf_tourney_wins")
    ct_wins = ct_wins.rename(columns={"WTeamID": "TeamID"})

    # Count conf tourney losses
    ct_losses = conf_tourney.groupby(["Season", "LTeamID"]).size().reset_index(name="conf_tourney_losses")
    ct_losses = ct_losses.rename(columns={"LTeamID": "TeamID"})

    # Merge
    result = participants.drop_duplicates(subset=["Season", "TeamID"])
    result = result.merge(conf_champs[["Season", "TeamID", "conf_tourney_champ"]],
                          on=["Season", "TeamID"], how="left")
    result = result.merge(ct_wins, on=["Season", "TeamID"], how="left")
    result = result.merge(ct_losses, on=["Season", "TeamID"], how="left")

    result["conf_tourney_champ"] = result["conf_tourney_champ"].fillna(0)
    result["conf_tourney_wins"] = result["conf_tourney_wins"].fillna(0)
    result["conf_tourney_losses"] = result["conf_tourney_losses"].fillna(0)

    return result[["Season", "TeamID", "conf_tourney_champ", "conf_tourney_wins", "conf_tourney_losses"]]


# ============================================================================
# ASSEMBLY: Merge all team-season features
# ============================================================================

def build_team_season_features(data, day_cutoff=133):
    """
    Master function: build all team-season level features.

    Args:
        data: dict of DataFrames from load_all_data()
        day_cutoff: only use games up to this DayNum (133 = pre-tournament)

    Returns:
        DataFrame with columns: Season, TeamID, feature1, feature2, ...
    """
    from . import data_loader

    print("[1/5] Loading and combining detailed results...")
    detailed = data_loader.load_regular_season_detailed(data)
    compact = data_loader.load_regular_season_compact(data)

    print("[2/5] Computing basic stats & team games...")
    team_games = compute_basic_stats(detailed)
    basic = aggregate_season_stats(team_games, day_cutoff=day_cutoff)

    print("[3/5] Computing advanced stats (Four Factors, efficiency)...")
    advanced = compute_advanced_stats(team_games, day_cutoff=day_cutoff)

    print("[4/5] Computing Strength of Schedule...")
    sos = compute_sos(team_games, basic, day_cutoff=day_cutoff)

    print("[5/5] Computing conference tournament features...")
    conf_tourney = data_loader.load_conference_tourney(data)
    conferences = data_loader.load_conferences(data)
    conf_feats = compute_conf_tourney_features(conf_tourney, conferences)

    # --- Merge everything ---
    print("Merging all features...")
    features = basic.copy()
    features = features.merge(advanced, on=["Season", "TeamID"], how="left")
    features = features.merge(sos, on=["Season", "TeamID"], how="left")
    features = features.merge(conf_feats, on=["Season", "TeamID"], how="left")

    # Fill NaN for conf tourney (teams that didn't play in it)
    for col in ["conf_tourney_champ", "conf_tourney_wins", "conf_tourney_losses"]:
        features[col] = features[col].fillna(0)

    print(f"Final team-season features: {features.shape[0]} rows, {features.shape[1]} columns")
    return features, team_games


if __name__ == "__main__":
    from data_loader import load_all_data
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    data = load_all_data()
    features, team_games = build_team_season_features.__wrapped__(data) if hasattr(build_team_season_features, '__wrapped__') else None, None
    print("Done!")
