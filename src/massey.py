"""
NCAA March Madness 2026 - Massey Ordinals Feature Engineering
Extract ranking features from 5.8M rows of ranking data (men only).
"""

import pandas as pd
import numpy as np


# Top ranking systems known to be predictive
TOP_SYSTEMS = ["POM", "SAG", "MOR", "DOL", "COL", "RPI", "AP", "USA", "WOL", "RTH",
               "WLK", "DOK", "ARG", "MAS", "SEL", "BPI", "KPK", "REW", "LOG", "SPR"]


def compute_massey_features(massey_df, day_cutoff=133):
    """
    Extract ranking features from Massey Ordinals.

    Strategy:
    1. Use final pre-tournament rankings (RankingDayNum=133 or closest)
    2. Aggregate across multiple ranking systems
    3. Compute ranking trends

    Args:
        massey_df: MMasseyOrdinals DataFrame (~5.8M rows)
        day_cutoff: max RankingDayNum to use (133 = right before tournament)

    Returns:
        DataFrame: (Season, TeamID) -> ranking features
    """
    df = massey_df.copy()

    # ===== 1. Final rankings (DayNum <= day_cutoff) =====
    print("  [Massey 1/4] Getting final rankings per system...")
    # Get the latest ranking for each (Season, SystemName, TeamID) before cutoff
    df_cut = df[df["RankingDayNum"] <= day_cutoff]

    # For each system+team+season, take the ranking with the highest DayNum
    idx = df_cut.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].idxmax()
    final_ranks = df_cut.loc[idx, ["Season", "SystemName", "TeamID", "OrdinalRank", "RankingDayNum"]]

    # ===== 2. Aggregate across all systems =====
    print("  [Massey 2/4] Aggregating across all ranking systems...")
    all_sys_agg = final_ranks.groupby(["Season", "TeamID"]).agg(
        rank_mean_all=("OrdinalRank", "mean"),
        rank_median_all=("OrdinalRank", "median"),
        rank_min_all=("OrdinalRank", "min"),
        rank_max_all=("OrdinalRank", "max"),
        rank_std_all=("OrdinalRank", "std"),
        num_systems=("OrdinalRank", "count"),
    ).reset_index()

    # ===== 3. Top systems individual + aggregate =====
    print("  [Massey 3/4] Processing top ranking systems...")
    top_ranks = final_ranks[final_ranks["SystemName"].isin(TOP_SYSTEMS)]

    # Pivot: each system as a column
    if len(top_ranks) > 0:
        pivoted = top_ranks.pivot_table(
            index=["Season", "TeamID"],
            columns="SystemName",
            values="OrdinalRank",
            aggfunc="first"
        ).reset_index()
        # Rename columns
        pivoted.columns = [f"rank_{c}" if c not in ["Season", "TeamID"] else c
                           for c in pivoted.columns]

        # Aggregate of top systems
        top_sys_agg = top_ranks.groupby(["Season", "TeamID"]).agg(
            rank_mean_top=("OrdinalRank", "mean"),
            rank_median_top=("OrdinalRank", "median"),
            rank_min_top=("OrdinalRank", "min"),
            rank_std_top=("OrdinalRank", "std"),
        ).reset_index()

        all_sys_agg = all_sys_agg.merge(top_sys_agg, on=["Season", "TeamID"], how="left")
        all_sys_agg = all_sys_agg.merge(pivoted, on=["Season", "TeamID"], how="left")

    # ===== 4. Ranking trend (mid-season vs end-of-season) =====
    print("  [Massey 4/4] Computing ranking trends...")
    # Mid-season: around DayNum 80 (roughly halfway)
    mid_season = df[(df["RankingDayNum"] >= 70) & (df["RankingDayNum"] <= 90)]
    if len(mid_season) > 0:
        mid_idx = mid_season.groupby(["Season", "SystemName", "TeamID"])["RankingDayNum"].idxmax()
        mid_ranks = mid_season.loc[mid_idx]

        mid_agg = mid_ranks.groupby(["Season", "TeamID"]).agg(
            rank_mean_mid=("OrdinalRank", "mean"),
        ).reset_index()

        all_sys_agg = all_sys_agg.merge(mid_agg, on=["Season", "TeamID"], how="left")

        # Trend: improvement = mid_rank - final_rank (positive = improved)
        all_sys_agg["rank_trend"] = all_sys_agg["rank_mean_mid"] - all_sys_agg["rank_mean_all"]
    else:
        all_sys_agg["rank_mean_mid"] = np.nan
        all_sys_agg["rank_trend"] = np.nan

    # ===== 5. Percentile rank (normalized) =====
    # Normalize by total teams in each season
    season_max = all_sys_agg.groupby("Season")["rank_mean_all"].max().reset_index()
    season_max.columns = ["Season", "max_rank"]
    all_sys_agg = all_sys_agg.merge(season_max, on="Season", how="left")
    all_sys_agg["rank_percentile"] = 1 - (all_sys_agg["rank_mean_all"] / all_sys_agg["max_rank"])
    all_sys_agg.drop(columns=["max_rank"], inplace=True)

    print(f"  Massey features: {all_sys_agg.shape[0]} team-seasons, "
          f"{all_sys_agg.shape[1] - 2} features")
    return all_sys_agg


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all_data, load_massey_ordinals

    data = load_all_data()
    massey = load_massey_ordinals(data)
    print(f"\nMassey Ordinals: {len(massey)} rows")
    print(f"Seasons: {massey.Season.min()}-{massey.Season.max()}")
    print(f"Systems: {massey.SystemName.nunique()}")

    feats = compute_massey_features(massey)
    print(f"\nResult shape: {feats.shape}")
    print(feats.head())
