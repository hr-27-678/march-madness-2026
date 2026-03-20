"""
NCAA March Madness 2026 - Feature Validation with LightGBM + Time-Series CV
Quick baseline to verify feature quality.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
import sys
import time
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_feature_pipeline


def time_series_cv(matchup_df, feature_cols, val_seasons=None, verbose=True):
    """
    Time-series cross-validation: train on past, validate on each season's tournament.

    Args:
        matchup_df: DataFrame with features + Label
        feature_cols: list of feature column names
        val_seasons: list of seasons to use as validation folds

    Returns:
        dict of results per fold and overall
    """
    if val_seasons is None:
        # Default: validate on 2019, 2021-2025 (skip 2020 = COVID)
        val_seasons = [2019, 2021, 2022, 2023, 2024, 2025]

    results = []
    all_preds = []
    all_labels = []

    for val_season in val_seasons:
        train_mask = matchup_df["Season"] < val_season
        val_mask = matchup_df["Season"] == val_season

        if val_mask.sum() == 0:
            continue

        X_train = matchup_df.loc[train_mask, feature_cols].copy()
        y_train = matchup_df.loc[train_mask, "Label"].copy()
        X_val = matchup_df.loc[val_mask, feature_cols].copy()
        y_val = matchup_df.loc[val_mask, "Label"].copy()

        # --- LightGBM ---
        lgb_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

        model = lgb.train(
            lgb_params,
            train_set,
            num_boost_round=1000,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        pred = model.predict(X_val)

        # Clip predictions
        pred = np.clip(pred, 0.02, 0.98)

        brier = brier_score_loss(y_val, pred)
        ll = log_loss(y_val, pred)

        results.append({
            "season": val_season,
            "n_games": len(y_val),
            "brier_score": brier,
            "log_loss": ll,
            "pred_mean": pred.mean(),
            "label_mean": y_val.mean(),
        })

        all_preds.extend(pred.tolist())
        all_labels.extend(y_val.tolist())

        if verbose:
            print(f"  Season {val_season}: Brier={brier:.4f}  LogLoss={ll:.4f}  "
                  f"n={len(y_val)}  pred_mean={pred.mean():.3f}")

    # Overall
    overall_brier = brier_score_loss(all_labels, all_preds)
    overall_ll = log_loss(all_labels, all_preds)

    if verbose:
        print(f"\n  OVERALL: Brier={overall_brier:.4f}  LogLoss={overall_ll:.4f}  "
              f"n={len(all_labels)}")

    # Get feature importance from last fold
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    return {
        "per_season": pd.DataFrame(results),
        "overall_brier": overall_brier,
        "overall_logloss": overall_ll,
        "feature_importance": importance,
        "model": model,
    }


def logistic_baseline(matchup_df, feature_cols, val_seasons=None, verbose=True):
    """Simple logistic regression baseline using seed diff + elo diff only."""
    if val_seasons is None:
        val_seasons = [2019, 2021, 2022, 2023, 2024, 2025]

    # Use only seed and elo features
    simple_cols = [c for c in feature_cols if any(
        k in c for k in ["diff_seed", "diff_elo_standard", "diff_win_rate",
                          "diff_net_efficiency_mean", "diff_rank_mean_all"]
    )]
    if not simple_cols:
        simple_cols = feature_cols[:5]

    if verbose:
        print(f"  Using {len(simple_cols)} features: {simple_cols}")

    all_preds = []
    all_labels = []
    results = []

    for val_season in val_seasons:
        train_mask = matchup_df["Season"] < val_season
        val_mask = matchup_df["Season"] == val_season

        if val_mask.sum() == 0:
            continue

        X_train = matchup_df.loc[train_mask, simple_cols].fillna(0)
        y_train = matchup_df.loc[train_mask, "Label"]
        X_val = matchup_df.loc[val_mask, simple_cols].fillna(0)
        y_val = matchup_df.loc[val_mask, "Label"]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_train_s, y_train)
        pred = lr.predict_proba(X_val_s)[:, 1]
        pred = np.clip(pred, 0.02, 0.98)

        brier = brier_score_loss(y_val, pred)
        results.append({"season": val_season, "brier": brier, "n": len(y_val)})
        all_preds.extend(pred.tolist())
        all_labels.extend(y_val.tolist())

        if verbose:
            print(f"  Season {val_season}: Brier={brier:.4f}  n={len(y_val)}")

    overall_brier = brier_score_loss(all_labels, all_preds)
    if verbose:
        print(f"\n  LR OVERALL: Brier={overall_brier:.4f}")

    return overall_brier


def seed_only_baseline(matchup_df, verbose=True):
    """Baseline using only historical seed matchup win rates."""
    # Historical seed matchup probabilities
    has_seed = matchup_df.dropna(subset=["diff_seed"])

    if len(has_seed) == 0:
        print("  No seed data available")
        return None

    val_seasons = [2019, 2021, 2022, 2023, 2024, 2025]
    all_preds = []
    all_labels = []

    for val_season in val_seasons:
        train = has_seed[has_seed["Season"] < val_season]
        val = has_seed[has_seed["Season"] == val_season]

        if len(val) == 0:
            continue

        # Build seed-diff -> win rate lookup
        seed_diff_wr = train.groupby("diff_seed")["Label"].mean().to_dict()

        preds = []
        for _, row in val.iterrows():
            sd = row["diff_seed"]
            if sd in seed_diff_wr:
                preds.append(seed_diff_wr[sd])
            else:
                # Interpolate: negative seed diff = higher seed = stronger
                preds.append(0.5)

        preds = np.clip(preds, 0.02, 0.98)
        brier = brier_score_loss(val["Label"], preds)

        all_preds.extend(preds)
        all_labels.extend(val["Label"].tolist())

        if verbose:
            print(f"  Season {val_season}: Brier={brier:.4f}  n={len(val)}")

    if all_preds:
        overall = brier_score_loss(all_labels, all_preds)
        if verbose:
            print(f"\n  SEED-ONLY OVERALL: Brier={overall:.4f}")
        return overall
    return None


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    t0 = time.time()

    # Run feature pipeline
    print("=" * 70)
    print("RUNNING FEATURE PIPELINE")
    print("=" * 70)
    matchup_df, feature_cols, team_features = run_feature_pipeline(
        target="train", use_massey=True
    )

    # ---- Baseline 1: Seed-only ----
    print("\n" + "=" * 70)
    print("BASELINE 1: SEED-ONLY (historical seed matchup probabilities)")
    print("=" * 70)
    seed_brier = seed_only_baseline(matchup_df)

    # ---- Baseline 2: Logistic Regression ----
    print("\n" + "=" * 70)
    print("BASELINE 2: LOGISTIC REGRESSION (seed + elo + key stats)")
    print("=" * 70)
    lr_brier = logistic_baseline(matchup_df, feature_cols)

    # ---- Main Model: LightGBM ----
    print("\n" + "=" * 70)
    print("MAIN MODEL: LightGBM with ALL features (Time-Series CV)")
    print("=" * 70)
    lgb_results = time_series_cv(matchup_df, feature_cols)

    # ---- Summary ----
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"VALIDATION SUMMARY (elapsed: {elapsed:.0f}s)")
    print("=" * 70)
    print(f"  Naive (all 0.5):       Brier = 0.2500")
    if seed_brier:
        print(f"  Seed-only baseline:    Brier = {seed_brier:.4f}")
    print(f"  Logistic Regression:   Brier = {lr_brier:.4f}")
    print(f"  LightGBM (all feats):  Brier = {lgb_results['overall_brier']:.4f}")

    print(f"\n  LightGBM per-season results:")
    print(lgb_results["per_season"].to_string(index=False))

    print(f"\n  Top 30 features by importance (gain):")
    print(lgb_results["feature_importance"].head(30).to_string(index=False))

    # Save
    out = Path(__file__).parent.parent / "validation_results.txt"
    with open(out, "w") as f:
        f.write(f"Naive (all 0.5):       Brier = 0.2500\n")
        if seed_brier:
            f.write(f"Seed-only baseline:    Brier = {seed_brier:.4f}\n")
        f.write(f"Logistic Regression:   Brier = {lr_brier:.4f}\n")
        f.write(f"LightGBM (all feats):  Brier = {lgb_results['overall_brier']:.4f}\n")
        f.write(f"\nPer-season:\n{lgb_results['per_season'].to_string()}\n")
        f.write(f"\nTop features:\n{lgb_results['feature_importance'].head(50).to_string()}\n")
    print(f"\n  Results saved to {out}")
