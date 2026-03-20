"""
NCAA March Madness 2026 - Elo Rating System
Implements multiple Elo variants for both men's and women's teams.
"""

import pandas as pd
import numpy as np


class EloSystem:
    """
    Enhanced Elo rating system with margin-of-victory adjustment.

    Features:
    - Season-to-season mean reversion
    - Home court advantage
    - Margin-of-victory multiplier (MOV)
    - Separate offensive/defensive Elo (optional)
    """

    def __init__(self, k=32, home_adv=100, mean_reversion=0.75, initial_elo=1500):
        self.k = k
        self.home_adv = home_adv
        self.mean_reversion = mean_reversion
        self.initial_elo = initial_elo
        self.ratings = {}  # (season, team_id) -> elo

    def _expected(self, elo_a, elo_b):
        """Expected score for team A."""
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))

    def _mov_multiplier(self, margin, elo_diff):
        """Margin-of-victory multiplier to weight blowouts."""
        return np.log(abs(margin) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))

    def get_elo(self, season, team_id):
        """Get current Elo for a team in a season."""
        return self.ratings.get((season, team_id), self.initial_elo)

    def _init_season(self, season, team_id, prev_season):
        """Initialize Elo for a new season with mean reversion."""
        if (prev_season, team_id) in self.ratings:
            prev_elo = self.ratings[(prev_season, team_id)]
            new_elo = self.mean_reversion * prev_elo + (1 - self.mean_reversion) * self.initial_elo
        else:
            new_elo = self.initial_elo
        self.ratings[(season, team_id)] = new_elo

    def update(self, season, w_team, l_team, w_score, l_score, w_loc, update_ratings=True):
        """
        Process a single game result and update ratings.

        Returns: (winner_elo_before, loser_elo_before, winner_elo_after, loser_elo_after)
        """
        elo_w = self.get_elo(season, w_team)
        elo_l = self.get_elo(season, l_team)

        # Adjust for home court
        if w_loc == "H":
            elo_w_adj = elo_w + self.home_adv
            elo_l_adj = elo_l
        elif w_loc == "A":
            elo_w_adj = elo_w
            elo_l_adj = elo_l + self.home_adv
        else:
            elo_w_adj = elo_w
            elo_l_adj = elo_l

        # Expected scores
        exp_w = self._expected(elo_w_adj, elo_l_adj)
        exp_l = 1 - exp_w

        # Margin-of-victory multiplier
        margin = w_score - l_score
        elo_diff = abs(elo_w_adj - elo_l_adj)
        mov = self._mov_multiplier(margin, elo_diff)

        # Update
        if update_ratings:
            new_elo_w = elo_w + self.k * mov * (1 - exp_w)
            new_elo_l = elo_l + self.k * mov * (0 - exp_l)
            self.ratings[(season, w_team)] = new_elo_w
            self.ratings[(season, l_team)] = new_elo_l
            return elo_w, elo_l, new_elo_w, new_elo_l
        else:
            return elo_w, elo_l, elo_w, elo_l

    def compute_all_elos(self, compact_results, day_cutoff=133):
        """
        Process all games chronologically and compute Elo ratings.
        Only updates using games up to day_cutoff.

        Args:
            compact_results: DataFrame with Season, DayNum, WTeamID, LTeamID, WScore, LScore, WLoc
            day_cutoff: max DayNum for updating ratings

        Returns:
            DataFrame: (Season, TeamID) -> final Elo before tournament
        """
        df = compact_results.sort_values(["Season", "DayNum"]).copy()

        # Get all seasons
        seasons = sorted(df["Season"].unique())

        elo_records = []

        for i, season in enumerate(seasons):
            prev_season = seasons[i - 1] if i > 0 else None

            # Initialize season Elos with mean reversion
            season_teams = set(
                df[df["Season"] == season]["WTeamID"].tolist() +
                df[df["Season"] == season]["LTeamID"].tolist()
            )
            for team_id in season_teams:
                if prev_season:
                    self._init_season(season, team_id, prev_season)
                else:
                    self.ratings[(season, team_id)] = self.initial_elo

            # Process games in chronological order
            season_games = df[df["Season"] == season].sort_values("DayNum")
            for _, game in season_games.iterrows():
                should_update = game["DayNum"] <= day_cutoff
                self.update(
                    season, game["WTeamID"], game["LTeamID"],
                    game["WScore"], game["LScore"], game["WLoc"],
                    update_ratings=should_update
                )

            # Record end-of-regular-season Elos
            for team_id in season_teams:
                elo_records.append({
                    "Season": season,
                    "TeamID": team_id,
                    "Elo": self.get_elo(season, team_id),
                })

        return pd.DataFrame(elo_records)


def compute_elo_features(compact_results, day_cutoff=133):
    """
    Compute multiple Elo variants and return combined features.

    Variants:
    1. Standard Elo (K=32, home_adv=100)
    2. Aggressive Elo (K=48, higher reactivity)
    3. Conservative Elo (K=20, more stable)
    """
    variants = {
        "elo_standard": EloSystem(k=32, home_adv=100, mean_reversion=0.75),
        "elo_aggressive": EloSystem(k=48, home_adv=100, mean_reversion=0.60),
        "elo_conservative": EloSystem(k=20, home_adv=100, mean_reversion=0.85),
    }

    dfs = []
    for name, system in variants.items():
        print(f"  Computing {name}...")
        elo_df = system.compute_all_elos(compact_results, day_cutoff=day_cutoff)
        elo_df = elo_df.rename(columns={"Elo": name})
        dfs.append(elo_df)

    # Merge all variants
    result = dfs[0]
    for df in dfs[1:]:
        result = result.merge(df, on=["Season", "TeamID"], how="outer")

    # Derived features
    result["elo_mean"] = result[["elo_standard", "elo_aggressive", "elo_conservative"]].mean(axis=1)
    result["elo_std"] = result[["elo_standard", "elo_aggressive", "elo_conservative"]].std(axis=1)

    print(f"  Elo features: {result.shape[0]} team-seasons, {result.shape[1] - 2} features")
    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_all_data, load_regular_season_compact

    data = load_all_data()
    compact = load_regular_season_compact(data)
    print(f"\nCompact results: {len(compact)} games")

    elo_feats = compute_elo_features(compact)
    print(f"\nElo features shape: {elo_feats.shape}")
    print(elo_feats.head(10))
    print(f"\nElo stats:\n{elo_feats.describe()}")
