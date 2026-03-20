"""
NCAA March Madness 2026 - Master Feature Pipeline
Assembles all features and builds final matchup-level training data.
"""

import pandas as pd
import numpy as np
import time
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    load_all_data, load_regular_season_detailed, load_regular_season_compact,
    load_tourney_compact, load_tourney_detailed, load_seeds,
    load_massey_ordinals, load_coaches, load_conferences,
    load_conference_tourney, build_tourney_labels, parse_submission,
)
from src.features import (
    compute_basic_stats, aggregate_season_stats,
    compute_advanced_stats, compute_sos, compute_conf_tourney_features,
)
from src.elo import compute_elo_features
from src.massey import compute_massey_features
from src.context_features import (
    compute_seed_features, compute_coach_features,
    compute_conference_strength, compute_coach_tourney_history,
)

DATA_DIR = Path(__file__).parent.parent / "march-machine-learning-mania-2026"


# ============================================================================
# MATCHUP FEATURE BUILDER
# ============================================================================

def build_matchup_features(team_features, team_a_id, team_b_id, season, feature_cols):
    """
    Build matchup-level features for TeamA vs TeamB.
    TeamA should have the lower TeamID (submission format).

    Features: difference (A - B), ratio (A / B), and individual values.
    """
    a_feats = team_features[
        (team_features["Season"] == season) & (team_features["TeamID"] == team_a_id)
    ]
    b_feats = team_features[
        (team_features["Season"] == season) & (team_features["TeamID"] == team_b_id)
    ]

    if len(a_feats) == 0 or len(b_feats) == 0:
        return None

    a_vals = a_feats[feature_cols].iloc[0]
    b_vals = b_feats[feature_cols].iloc[0]

    result = {}
    for col in feature_cols:
        result[f"A_{col}"] = a_vals[col]
        result[f"B_{col}"] = b_vals[col]
        result[f"diff_{col}"] = a_vals[col] - b_vals[col]

    return result


def build_matchup_features_vectorized(matchups_df, team_features, feature_cols):
    """
    Vectorized version: build matchup features for all matchups at once.
    Much faster than row-by-row.

    Args:
        matchups_df: DataFrame with Season, TeamA, TeamB columns
        team_features: DataFrame with Season, TeamID, and feature columns
        feature_cols: list of feature column names to use

    Returns:
        DataFrame with matchup features aligned to matchups_df
    """
    # Rename for TeamA merge
    a_feats = team_features[["Season", "TeamID"] + feature_cols].copy()
    a_feats = a_feats.rename(columns={c: f"A_{c}" for c in feature_cols})
    a_feats = a_feats.rename(columns={"TeamID": "TeamA"})

    # Rename for TeamB merge
    b_feats = team_features[["Season", "TeamID"] + feature_cols].copy()
    b_feats = b_feats.rename(columns={c: f"B_{c}" for c in feature_cols})
    b_feats = b_feats.rename(columns={"TeamID": "TeamB"})

    # Merge
    result = matchups_df[["Season", "TeamA", "TeamB"]].copy()
    result = result.merge(a_feats, on=["Season", "TeamA"], how="left")
    result = result.merge(b_feats, on=["Season", "TeamB"], how="left")

    # Compute differences (all at once to avoid fragmentation)
    diff_data = {}
    for col in feature_cols:
        diff_data[f"diff_{col}"] = result[f"A_{col}"].values - result[f"B_{col}"].values
    diff_df = pd.DataFrame(diff_data, index=result.index)
    result = pd.concat([result, diff_df], axis=1)

    return result


# ============================================================================
# GLM TEAM QUALITY (Bradley-Terry)
# ============================================================================

def compute_glm_quality(compact_df):
    """
    Bradley-Terry team quality scores via weighted logistic regression.
    Each game is a pair (team_i, team_j): outcome=1 if team_i wins.
    Recent games (high DayNum) weighted 2x early games.
    Returns DataFrame with Season, TeamID, glm_quality.
    """
    from sklearn.linear_model import LogisticRegression as _LR
    from scipy.sparse import csr_matrix

    results = []
    for season in sorted(compact_df['Season'].unique()):
        df_s = compact_df[compact_df['Season'] == season].copy()
        if len(df_s) < 10:
            continue

        # Weight: recent games count more — DayNum/max → [0,1] → +1 → [1, 2]
        max_day = df_s['DayNum'].max()
        df_s['gw'] = df_s['DayNum'] / max(max_day, 1) + 1.0

        wins   = pd.DataFrame({'T1': df_s['WTeamID'].values, 'T2': df_s['LTeamID'].values,
                                'outcome': 1, 'w': df_s['gw'].values})
        losses = pd.DataFrame({'T1': df_s['LTeamID'].values, 'T2': df_s['WTeamID'].values,
                                'outcome': 0, 'w': df_s['gw'].values})
        df_long = pd.concat([wins, losses], ignore_index=True)

        teams = sorted(set(df_long['T1']) | set(df_long['T2']))
        if len(teams) < 3:
            continue
        team_idx = {t: i for i, t in enumerate(teams)}
        n, k = len(df_long), len(teams)

        # Bradley-Terry design: T1 column = +1, T2 column = -1 (no intercept)
        rows = np.repeat(np.arange(n), 2)
        cols = np.array([team_idx[t] for t in df_long['T1']] +
                        [team_idx[t] for t in df_long['T2']])
        vals = np.array([1.0] * n + [-1.0] * n)
        X_bt = csr_matrix((vals, (rows, cols)), shape=(n, k))

        try:
            model = _LR(C=1.0, max_iter=500, solver='lbfgs',
                        fit_intercept=False, random_state=42)
            model.fit(X_bt, df_long['outcome'].values, sample_weight=df_long['w'].values)
            coeffs = model.coef_[0]
            for team, idx in team_idx.items():
                results.append({'Season': season, 'TeamID': team, 'glm_quality': coeffs[idx]})
        except Exception:
            pass  # skip on convergence failure

    if not results:
        return pd.DataFrame(columns=['Season', 'TeamID', 'glm_quality'])
    return pd.DataFrame(results)


# ============================================================================
# HISTORICAL HEAD-TO-HEAD
# ============================================================================

def compute_h2h_features(compact_results, matchups_df, lookback_years=5):
    """
    Compute historical head-to-head record between teams.
    """
    df = compact_results.copy()
    records = []

    for _, matchup in matchups_df.iterrows():
        season = matchup["Season"]
        team_a = matchup["TeamA"]
        team_b = matchup["TeamB"]

        # Look at past games between these teams
        past = df[
            (df["Season"] >= season - lookback_years) &
            (df["Season"] < season)
        ]

        # Games where A won vs B
        a_wins = past[
            (past["WTeamID"] == team_a) & (past["LTeamID"] == team_b)
        ]
        # Games where B won vs A
        b_wins = past[
            (past["WTeamID"] == team_b) & (past["LTeamID"] == team_a)
        ]

        total = len(a_wins) + len(b_wins)
        records.append({
            "h2h_games": total,
            "h2h_a_wins": len(a_wins),
            "h2h_a_win_rate": len(a_wins) / total if total > 0 else 0.5,
            "h2h_avg_margin": (
                (a_wins["WScore"] - a_wins["LScore"]).sum() -
                (b_wins["WScore"] - b_wins["LScore"]).sum()
            ) / total if total > 0 else 0,
        })

    return pd.DataFrame(records)


# ============================================================================
# MASTER PIPELINE
# ============================================================================

def run_feature_pipeline(target="train", day_cutoff=133, use_massey=True):
    """
    Run the complete feature engineering pipeline.

    Args:
        target: "train" (build training data from historical tournaments)
                "stage1" (build Stage1 submission matchups)
                "stage2" (build Stage2 submission matchups)
        day_cutoff: DayNum cutoff for features (133 = pre-tournament)
        use_massey: whether to compute Massey features (slow, ~5.8M rows)

    Returns:
        matchup_df: DataFrame with all features + labels (for train) or IDs (for predict)
        feature_cols: list of feature column names
    """
    t0 = time.time()

    # ========== LOAD DATA ==========
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    data = load_all_data()

    detailed = load_regular_season_detailed(data)
    compact = load_regular_season_compact(data)
    tourney_compact = load_tourney_compact(data)
    seeds = load_seeds(data)
    team_conferences = load_conferences(data)
    conf_tourney = load_conference_tourney(data)

    # ========== TEAM-GAME FEATURES ==========
    print("\n" + "=" * 60)
    print("COMPUTING TEAM-SEASON FEATURES")
    print("=" * 60)

    # 1. Basic stats
    print("\n[1/7] Basic season statistics...")
    team_games = compute_basic_stats(detailed)
    basic_stats = aggregate_season_stats(team_games, day_cutoff=day_cutoff)
    print(f"  -> {basic_stats.shape[0]} team-seasons, {basic_stats.shape[1]} cols")

    # 2. Advanced stats
    print("\n[2/7] Advanced basketball statistics...")
    adv_stats = compute_advanced_stats(team_games, day_cutoff=day_cutoff)
    print(f"  -> {adv_stats.shape[1] - 2} features")

    # 3. SOS
    print("\n[3/7] Strength of Schedule...")
    sos = compute_sos(team_games, basic_stats, day_cutoff=day_cutoff)
    print(f"  -> {sos.shape[1] - 2} features")

    # 4. Elo
    print("\n[4/7] Elo Ratings (3 variants)...")
    elo_feats = compute_elo_features(compact, day_cutoff=day_cutoff)
    print(f"  -> {elo_feats.shape[1] - 2} features")

    # 4.5. GLM Team Quality (Bradley-Terry, recent-game weighted)
    print("\n[+/7] GLM Team Quality (Bradley-Terry)...")
    glm_m = compute_glm_quality(compact[compact['WTeamID'] < 2000])
    glm_w = compute_glm_quality(compact[compact['WTeamID'] >= 2000])
    glm_quality = pd.concat([glm_m, glm_w], ignore_index=True)
    print(f"  -> {len(glm_quality):,} team-season quality scores")

    # 5. Massey Ordinals (men only)
    massey_feats = None
    if use_massey:
        print("\n[5/7] Massey Ordinals Rankings (men only, this may take a minute)...")
        massey = load_massey_ordinals(data)
        massey_feats = compute_massey_features(massey, day_cutoff=day_cutoff)
        print(f"  -> {massey_feats.shape[1] - 2} features")
    else:
        print("\n[5/7] Skipping Massey Ordinals")

    # 6. Conference features
    print("\n[6/7] Conference & coach features...")
    conf_tourney_feats = compute_conf_tourney_features(conf_tourney, None)
    conf_strength = compute_conference_strength(team_conferences, basic_stats)

    # Coach features (men only, we only have MTeamCoaches)
    coaches = load_coaches(data)
    coach_feats = compute_coach_features(coaches)
    # Coach tournament history
    m_tourney = data["MNCAATourneyCompactResults"]
    coach_tourney = compute_coach_tourney_history(coaches, m_tourney)

    # 7. Seeds
    print("\n[7/7] Seed features...")
    seed_feats = compute_seed_features(seeds)

    # ========== MERGE ALL TEAM FEATURES ==========
    print("\n" + "=" * 60)
    print("MERGING ALL TEAM-SEASON FEATURES")
    print("=" * 60)

    team_features = basic_stats.copy()

    # Merge advanced
    team_features = team_features.merge(adv_stats, on=["Season", "TeamID"], how="left")

    # Merge SOS
    team_features = team_features.merge(sos, on=["Season", "TeamID"], how="left")

    # Merge Elo
    team_features = team_features.merge(elo_feats, on=["Season", "TeamID"], how="left")

    # Merge GLM Team Quality (complementary to Elo — global fit vs sequential update)
    team_features = team_features.merge(glm_quality, on=["Season", "TeamID"], how="left")

    # Merge Massey (men only - will be NaN for women)
    if massey_feats is not None:
        team_features = team_features.merge(massey_feats, on=["Season", "TeamID"], how="left")

    # Merge conference features
    conf_cols_to_use = [c for c in conf_strength.columns
                        if c not in ["Season", "TeamID", "ConfAbbrev", "Gender"]]
    team_features = team_features.merge(
        conf_strength[["Season", "TeamID"] + conf_cols_to_use],
        on=["Season", "TeamID"], how="left"
    )

    # Merge conference tourney features
    team_features = team_features.merge(
        conf_tourney_feats, on=["Season", "TeamID"], how="left"
    )
    for col in ["conf_tourney_champ", "conf_tourney_wins", "conf_tourney_losses"]:
        team_features[col] = team_features[col].fillna(0)

    # Merge coach features (men only)
    coach_cols = ["coach_career_year", "coach_tenure", "is_new_coach"]
    team_features = team_features.merge(
        coach_feats[["Season", "TeamID"] + coach_cols],
        on=["Season", "TeamID"], how="left"
    )
    team_features = team_features.merge(
        coach_tourney[["Season", "TeamID", "coach_tourney_apps"]],
        on=["Season", "TeamID"], how="left"
    )

    # Merge seeds (only for tournament teams)
    team_features = team_features.merge(
        seed_feats, on=["Season", "TeamID"], how="left"
    )

    print(f"\nTotal team features: {team_features.shape[0]} rows, {team_features.shape[1]} cols")

    # ========== DEFINE FEATURE COLUMNS ==========
    exclude_cols = {"Season", "TeamID", "Gender", "CoachName", "ConfAbbrev", "Region"}
    feature_cols = [c for c in team_features.columns if c not in exclude_cols]

    # ========== BUILD MATCHUP DATA ==========
    print("\n" + "=" * 60)
    print(f"BUILDING MATCHUP DATA (mode={target})")
    print("=" * 60)

    if target == "train":
        # Build labels from historical tournaments
        matchups = build_tourney_labels(tourney_compact)
        # Filter to seasons where we have detailed data (2003+)
        matchups = matchups[matchups["Season"] >= 2003].reset_index(drop=True)
        # Remove 2020 (COVID, no tournament)
        matchups = matchups[matchups["Season"] != 2020].reset_index(drop=True)
        print(f"  Training matchups: {len(matchups)} games ({matchups.Season.min()}-{matchups.Season.max()})")

    elif target == "stage1":
        sub = parse_submission(DATA_DIR / "SampleSubmissionStage1.csv")
        matchups = sub[["ID", "Season", "TeamA", "TeamB"]].copy()
        print(f"  Stage1 matchups: {len(matchups)}")

    elif target == "stage2":
        sub = parse_submission(DATA_DIR / "SampleSubmissionStage2.csv")
        matchups = sub[["ID", "Season", "TeamA", "TeamB"]].copy()
        print(f"  Stage2 matchups: {len(matchups)}")

    else:
        raise ValueError(f"Unknown target: {target}")

    # Build matchup features (vectorized)
    print("\n  Building matchup features (vectorized)...")
    matchup_feats = build_matchup_features_vectorized(matchups, team_features, feature_cols)

    # Merge back with matchup info
    if target == "train":
        matchup_feats["Label"] = matchups["Label"].values
        matchup_feats["ScoreDiff"] = matchups["ScoreDiff"].values
        matchup_feats["Gender"] = matchups["Gender"].values
    if "ID" in matchups.columns:
        matchup_feats["ID"] = matchups["ID"].values

    # Head-to-head features (skip for stage2 to save time)
    if target == "train":
        print("\n  Computing head-to-head features...")
        h2h = compute_h2h_features(compact, matchups, lookback_years=5)
        for col in h2h.columns:
            matchup_feats[col] = h2h[col].values

    # ========== ADDITIONAL INTERACTION FEATURES ==========
    matchup_feats['is_mens'] = (matchup_feats['TeamA'] < 2000).astype(int)

    if 'A_seed' in matchup_feats.columns:
        matchup_feats['seed_product'] = matchup_feats['A_seed'] * matchup_feats['B_seed']
        matchup_feats['seed_sum'] = matchup_feats['A_seed'] + matchup_feats['B_seed']
        matchup_feats['seed_abs_diff'] = matchup_feats['diff_seed'].abs()

    # ------ Ratio / interaction features ------
    # log-ratio is numerically stable (no div-by-zero) and symmetric around 0.
    # Cross-matchup ratios (A's offense vs B's defense) capture genuine matchup
    # dynamics that same-side differences miss.
    _EPS = 1e-6

    def _safe_log_ratio(a, b):
        return np.log(np.clip(a, _EPS, None) / np.clip(b, _EPS, None))

    # Win-rate log-ratio
    if 'A_win_rate' in matchup_feats.columns:
        matchup_feats['log_ratio_win_rate'] = _safe_log_ratio(
            matchup_feats['A_win_rate'], matchup_feats['B_win_rate'])

    # Scoring output log-ratio
    if 'A_avg_score' in matchup_feats.columns:
        matchup_feats['log_ratio_avg_score'] = _safe_log_ratio(
            matchup_feats['A_avg_score'], matchup_feats['B_avg_score'])

    # Elo log-ratio  (shift by 1500 so values are always positive)
    if 'A_elo_mean' in matchup_feats.columns:
        matchup_feats['log_ratio_elo'] = _safe_log_ratio(
            matchup_feats['A_elo_mean'] + 1500,
            matchup_feats['B_elo_mean'] + 1500)

    # Cross matchup: A's offense vs B's defense (and vice versa)
    if ('A_off_efficiency_mean' in matchup_feats.columns and
            'B_def_efficiency_mean' in matchup_feats.columns):
        matchup_feats['cross_a_off_b_def'] = _safe_log_ratio(
            matchup_feats['A_off_efficiency_mean'],
            matchup_feats['B_def_efficiency_mean'])
        matchup_feats['cross_b_off_a_def'] = _safe_log_ratio(
            matchup_feats['B_off_efficiency_mean'],
            matchup_feats['A_def_efficiency_mean'])
        # Net matchup edge: how much A's attack vs B's defense exceeds B's attack vs A's defense
        matchup_feats['net_matchup_eff'] = (
            matchup_feats['cross_a_off_b_def'] - matchup_feats['cross_b_off_a_def'])

    # Interaction: seed × Elo  (doubly-dominant matchups compound confidence)
    if 'diff_seed' in matchup_feats.columns and 'diff_elo_mean' in matchup_feats.columns:
        matchup_feats['interact_seed_x_elo'] = (
            matchup_feats['diff_seed'] * matchup_feats['diff_elo_mean'])

    # Interaction: seed × win-rate
    if 'diff_seed' in matchup_feats.columns and 'diff_win_rate' in matchup_feats.columns:
        matchup_feats['interact_seed_x_winrate'] = (
            matchup_feats['diff_seed'] * matchup_feats['diff_win_rate'])

    # Interaction: Elo × win-rate
    if 'diff_elo_mean' in matchup_feats.columns and 'diff_win_rate' in matchup_feats.columns:
        matchup_feats['interact_elo_x_winrate'] = (
            matchup_feats['diff_elo_mean'] * matchup_feats['diff_win_rate'])

    # ========== IDENTIFY DIFF FEATURES (main predictors) ==========
    diff_cols = [c for c in matchup_feats.columns if c.startswith("diff_")]
    h2h_cols  = [c for c in matchup_feats.columns if c.startswith("h2h_")]

    _base_extra = ['is_mens', 'seed_product', 'seed_sum', 'seed_abs_diff']
    _ratio_extra = [
        'log_ratio_win_rate', 'log_ratio_avg_score', 'log_ratio_elo',
        'cross_a_off_b_def', 'cross_b_off_a_def', 'net_matchup_eff',
        'interact_seed_x_elo', 'interact_seed_x_winrate', 'interact_elo_x_winrate',
    ]
    extra_cols = [c for c in _base_extra + _ratio_extra if c in matchup_feats.columns]
    all_feature_cols = diff_cols + h2h_cols + extra_cols

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 60}")
    print(f"  Matchup shape: {matchup_feats.shape}")
    print(f"  Feature columns: {len(all_feature_cols)}")
    print(f"  diff features: {len(diff_cols)}")
    print(f"  h2h features: {len(h2h_cols)}")

    if target == "train":
        print(f"  Label distribution: {matchup_feats['Label'].mean():.3f}")
        print(f"  Missing values: {matchup_feats[all_feature_cols].isnull().sum().sum()} "
              f"/ {matchup_feats[all_feature_cols].size}")

    return matchup_feats, all_feature_cols, team_features


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    print("Running feature pipeline (training mode)...")
    matchup_df, feature_cols, team_features = run_feature_pipeline(
        target="train", use_massey=True
    )

    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Matchup DataFrame: {matchup_df.shape}")
    print(f"Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols):
        print(f"  {i+1:3d}. {col}")

    # Save for inspection
    out_path = Path(__file__).parent.parent / "train_features.csv"
    matchup_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Quick stats
    print(f"\nFirst 5 rows of diff features:")
    diff_cols = [c for c in feature_cols if c.startswith("diff_")]
    print(matchup_df[diff_cols[:10]].head())
