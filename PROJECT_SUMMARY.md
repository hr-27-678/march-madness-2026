# NCAA March Madness 2026 Kaggle Competition - Project Summary

## Competition Overview

- **Competition**: March Machine Learning Mania 2026
- **Task**: Predict win probabilities for all possible matchups in the 2026 NCAA Basketball Tournament (Men + Women)
- **Metric**: Brier Score = mean((predicted - actual)^2), lower is better
- **Submission**: 132,133 rows (all possible team pairs), each with P(TeamA wins)
- **Final Score**: Public LB = 0.15242 (based on ~16 games evaluated at Day 1)
- **CV Score**: 0.1619 (6-fold time-series CV)

---

## Phase 1: Data Pipeline & Feature Engineering

### 1.1 Data Infrastructure (`src/`)

Built a modular feature engineering pipeline:

- **`data_loader.py`**: Load 35 CSV files (teams, seeds, results, Massey ordinals, etc.)
- **`features.py`**: 30 basic season stats + 63 advanced basketball metrics (eFG%, TO rate, ORB%, FT rate, tempo, etc.)
- **`elo.py`**: 3 Elo variants (standard/aggressive/conservative) + mean + std = 5 features
- **`massey.py`**: Massey Ordinals rankings aggregation (33 features, men only)
- **`context_features.py`**: Conference features, coach features, seed features
- **`pipeline.py`**: End-to-end pipeline: team features -> matchup features (diff_A - diff_B)

### 1.2 Feature Design (171 -> 19 features)

**Raw team features**: 156 columns per team-season

**Matchup features** (TeamA - TeamB differences):
- 154 diff_* features (basic stats, advanced stats, Elo, Massey rankings, SOS, etc.)
- 4 h2h features (head-to-head history) -- later removed
- 9 ratio/interaction features added manually:
  - `log_ratio_win_rate`, `log_ratio_avg_score`, `log_ratio_elo`
  - `cross_a_off_b_def`, `cross_b_off_a_def`, `net_matchup_eff`
  - `interact_seed_x_elo`, `interact_seed_x_winrate`, `interact_elo_x_winrate`

### 1.3 GLM Team Quality (Bradley-Terry)

Added a Bradley-Terry model to compute team quality scores from game results. Provides a theoretically grounded "power rating" complementing Elo. **Note**: GLM quality was added to the 156-feature pool but did not rank in the top-19 by LGB importance, so it is not in the final model. It may have contributed indirectly by being correlated with features that were selected (e.g., Elo, win rate).

### 1.4 Regular Season Data Augmentation

- Tournament-only: 2,851 games (2003-2025)
- Regular season: 234,238 games
- Combined with TOURNEY_WEIGHT=5.0 (tournament games weighted 5x)
- **Key finding**: Augmentation helps men's CatBoost but hurts men's LGB and all women's models
- **Decision**: Men's CatBoost uses augmented data, everything else uses tournament-only

---

## Phase 2: Model Development & Iteration

### 2.1 Baseline Models

| Model | Brier | Notes |
|-------|-------|-------|
| Naive (0.5) | 0.2500 | Predict 50% for everything |
| Seed-only | 0.1770 | Historical seed win rates |
| Logistic Regression (7 features) | 0.1661 | Simple but strong baseline |
| LGB (all 171 features) | 0.1698 | Worse than LR -- overfitting on 171 features |

**Key insight**: More features != better. LGB with all features was worse than 7-feature LR.

### 2.2 Feature Selection Journey

Feature selection was a major focus:

1. **Initial approach**: Tried best_k optimization over {20, 50, 80} -- unstable across runs
2. **Teammate's finding**: best_k=20 gave Brier 0.1568 (vs our 0.1599 with k=50)
3. **Final decision**: Fixed at top-20 by LGB gain importance
4. **h2h removal**: h2h_avg_margin was #5 by importance but completely useless in Stage 2 (all zeros for future games). Removing it was the single most important generalization fix.
5. **Feature stability check**: Verified all 19 remaining features appear in top-20 across all 6 CV folds (6/6 stability)

**Final 19 features**:
1. diff_elo_conservative
2. diff_seed
3. diff_elo_standard
4. diff_sos_mean
5. diff_elo_std
6. interact_seed_x_elo
7. diff_opp_ft_pct_std
8. diff_opp_ft_pct_mean
9. diff_efg_trend
10. diff_FT_rate_std
11. diff_ft_pct_std
12. diff_sos_median
13. diff_conf_median_win_rate
14. diff_ast_to_ratio_std
15. diff_blk_rate_std
16. diff_fg3_pct_std
17. diff_stl_rate_std
18. diff_avg_margin
19. diff_reb_margin_std

### 2.3 Models Tried and Removed

Started with 6 models, simplified down to 2:

| Model | Status | Reason |
|-------|--------|--------|
| LightGBM | **Kept** | Core model, tuned with Optuna |
| CatBoost | **Kept** | depth=4 (conservative), adds +0.0047 via equal-weight ensemble |
| XGBoost | Removed | Redundant with LGB, added complexity |
| Logistic Regression | Removed | Useful as baseline but doesn't help ensemble |
| MLP | Removed | Weight 0.417 in optimized ensemble was suspicious overfitting artifact |
| Spread Regression | Removed | Brier 0.2229 (worse than seed baseline), correlation 0.843 with LGB |

### 2.4 Spread Regression Experiment

Implemented XGB regression predicting score margin -> IsotonicRegression mapping to probabilities:
- Men-only, tournament data
- **Result**: Brier 0.2229 (terrible, worse than seed-only 0.1770)
- Correlation with LGB: 0.843 (passed < 0.90 threshold)
- **Decision**: Not added despite low correlation -- absolute performance too poor, would drag ensemble down

---

## Phase 3: Overfitting Diagnosis & Simplification

### 3.1 The Overfitting Problem

Key observation: 795 OOF tournament samples had to support all model decisions. Original pipeline had:
- 9 Optuna hyperparameters
- 5-6 ensemble weight parameters (optimized on validation)
- Clipping parameters
- Seed-prior blending parameter
- Temperature parameter
- **Total: 15+ free parameters on 795 samples = 53:1 ratio (dangerous)**

Evidence of overfitting:
- MLP weight = 0.417 in optimized ensemble (implausibly high for a weak model)
- Spread weight = 0.752 (artifact of optimizing on small sample)
- Optuna gain varied 0.001-0.003 across different seeds
- TOURNEY_WEIGHT grid search: best=15 for proxy LGB but hurt full pipeline

### 3.2 Simplification Strategy

**Philosophy**: "Free parameter budget" -- every optimized parameter costs validation samples

Changes made:
1. **Removed 4 models** (XGB, LR, MLP, Spread) -- zero ensemble weights to optimize
2. **Fixed equal weights** (LGB + CB at 0.5/0.5) -- 0 free parameters
3. **Reduced Optuna** from 9 to 5 parameters, 100 trials, fixed seed=42
4. **Fixed Optuna search space**: num_leaves (4,16) to match max_depth (3,5)
5. **Removed clipping optimization** -- fixed at (0.02, 0.98)
6. **Removed seed-prior blending** -- direction was wrong
7. **Temperature scaling only** -- single parameter calibration

**After simplification**: 5 (Optuna) + 0 (weights) + 1 (temperature) = **6 free parameters / 795 samples = 132:1 ratio**

### 3.3 Optuna Tuning

- 5 parameters: num_leaves, max_depth, min_child_samples, feature_fraction, reg_lambda
- Fixed: learning_rate=0.05, bagging_fraction=0.8, bagging_freq=1, reg_alpha=0
- 100 trials, TPESampler(seed=42)
- **Result**: Exp0 (default) 0.1674 -> Exp1 (tuned) 0.1637, gain = 0.0037
- Best params: num_leaves=5, max_depth=3 (very conservative -- model prefers simplicity)

### 3.4 Multi-Seed Ensemble

LGB trained with 3 seeds (42, 43, 44) and averaged:
- Zero additional free parameters
- Reduces single-seed randomness
- Especially valuable since Optuna gain (0.003) is same order as seed variation

---

## Phase 4: Gender-Specific Modeling

### 4.1 Why Gender-Specific

- Men: 1,449 tournament games, have Massey Ordinals (33 extra features)
- Women: 1,402 tournament games, NO Massey Ordinals
- Different dynamics: women's tournament more seed-predictable

### 4.2 Gender Results

| Component | Men | Women |
|-----------|-----|-------|
| LGB | 0.1873 | 0.1390 |
| CatBoost | 0.1870 (aug) | 0.1380 |
| LGB+CB equal | 0.1844 | 0.1373 |
| Augmentation | CB only, LGB hurts | All hurts |

Women's model is much stronger (0.137 vs 0.184 for men). Women's games being more seed-predictable is the main reason.

---

## Phase 5: Calibration & Post-Processing

### 5.1 Temperature Scaling

- Optimal temperature: 0.90 (pushes predictions toward extremes)
- Brier improvement: 0.1610 -> 0.1608 (minimal)
- **Why T < 1.0 is correct**: For Brier Score, being more confident when correct saves more than the penalty for being wrong, when base accuracy is high

### 5.2 Approaches Tried and Removed

- **Seed-prior clipping**: Overriding predictions based on seed matchup history -- removed (direction was wrong, T<1 already handles this)
- **Dynamic clip range**: Removed, fixed at (0.02, 0.98)
- **Isotonic regression calibration**: Not used (temperature is simpler and sufficient)

---

## Phase 6: Validation Framework

### 6.1 Time-Series CV

6 folds, each using all prior seasons for training:
- Fold 0: Train <= 2018, Val = 2019 (130 games)
- Fold 1: Train <= 2020, Val = 2021 (129 games)
- Fold 2-5: 2022-2025 (134 games each)
- Total validation: 795 games
- **Note**: 2020 skipped (COVID, no tournament)

### 6.2 Key Metrics Tracked

- Brier Score (primary)
- Per-fold Brier (consistency check)
- LGB-CB OOF correlation (ensemble diversity: Men=0.956, Women=0.976)
- ECE (Expected Calibration Error): 0.035
- Sign test: require 6/6 folds improving (p=0.016) to trust a change

### 6.3 Sanity Checks

- Prediction distribution (mean~0.50, std~0.35)
- Symmetry check: P(A>B) + P(B>A) ~ 1.0
- Training label balance (0.505, near-perfect)
- Stage1 backtest (optimistic but useful for debugging)

---

## Phase 7: Key Decisions & Reasoning

### 7.1 h2h Feature Removal (Most Impactful Decision)

- h2h_avg_margin was #5 feature by importance (gain=445)
- In training CV: real head-to-head data exists
- In Stage 2 prediction: ALL h2h values = 0 (future games have no history)
- **This is classic train/inference distribution shift**
- Removing it made CV Brier slightly worse (0.1608 -> 0.1619) but improved generalization
- Optuna gain actually increased after removal (0.0027 -> 0.0037), suggesting h2h was interfering with hyperparameter optimization

### 7.2 Temperature Direction (T < 1.0 vs T > 1.0)

Mathematical analysis showed T < 1.0 (aggressive) is correct for Brier Score:
- For a 97% correct prediction: Brier = 0.0009 (aggressive) vs 0.04 (conservative at 80%)
- Expected cost of occasional upset at 5% rate: aggressive still wins
- Confirmed by paris-madness-2023 gold solution approach (extreme seed overrides)

### 7.3 Not Adding Spread Despite Low Correlation

- Spread-LGB correlation = 0.843 (< 0.90 threshold, should add)
- But Spread Brier = 0.2229 (worse than seed-only baseline)
- Equal-weight inclusion would pull men's Brier from ~0.187 toward ~0.200
- **Lesson**: Low correlation is necessary but not sufficient; absolute quality matters too

### 7.4 Augmentation Strategy

- Regular season data (234K games) is 82x more than tournament (2.8K)
- Grid search on TOURNEY_WEIGHT showed 15.0 best for proxy LGB but hurt full pipeline
- Fixed at 5.0 (reasonable compromise)
- Gender-specific: only men's CatBoost benefits from augmentation
- **Lesson**: More data isn't always better; distribution shift between RS and tournament matters

---

## Phase 8: Past Solution Analysis

### Studied Solutions

1. **paris-madness-2023** (Gold): Spread regression, seed overrides, aggressive predictions
2. **Other top solutions**: GLM quality, Cauchy loss, dynamic seed clipping

### What We Adopted

- GLM Team Quality (Bradley-Terry) -- added to features
- Spread regression + Isotonic -- tested but rejected (Brier too poor)
- Aggressive temperature (T < 1.0) -- aligned with gold solution philosophy

### What We Rejected

- Seed overrides (too many free parameters for uncertain gain)
- Cauchy loss / Huber loss for spread regression (spread model itself was too weak)
- Complex stacking (Ridge meta-learner on OOF predictions -- removed for simplicity)

---

## Final Architecture

```
Input: 35 CSV files (2003-2025 seasons)
  |
  v
Feature Pipeline (src/pipeline.py)
  - Basic stats (30) + Advanced (63) + SOS (7) + Elo (5) + GLM (1)
  - Massey Ordinals (33, men only)
  - Conference/Coach/Seed features
  = 156 team-level features -> 171 matchup diff features
  |
  v
Feature Selection: Top-19 by LGB gain importance
  (h2h removed, all 19 stable across 6/6 folds)
  |
  v
Gender Split: Men (1,449 games) | Women (1,402 games)
  |
  v
Optuna Tuning: 5 params, 100 trials, seed=42
  -> num_leaves=5, max_depth=3 (conservative)
  |
  v
Models:
  - LGB x 3 seeds (42, 43, 44), averaged
  - CatBoost (depth=4, conservative)
  - Men: CB uses augmented data (tourney + RS, weight=5)
  - Women: tournament-only
  |
  v
Equal-Weight Ensemble: 0.5 * LGB + 0.5 * CB (0 free parameters)
  |
  v
Temperature Scaling: T=0.90 (1 free parameter)
  |
  v
Clip: (0.02, 0.98)
  |
  v
Submission: 132,133 predictions
```

**Total free parameters**: 5 (Optuna) + 0 (weights) + 1 (temperature) = **6**
**Parameter/sample ratio**: 6 / 795 = **1:132** (very healthy)

---

## Results Timeline

| Version | CV Brier | Key Change | Notes |
|---------|----------|------------|-------|
| Seed-only baseline | 0.1770 | -- | |
| LR (7 features) | 0.1661 | Strong simple baseline | |
| LGB (171 features) | 0.1698 | Overfitting on too many features | |
| LGB (top-20) default | 0.1674 | Feature selection helps | |
| + Optuna tuning | 0.1637 | Hyperparameter optimization | |
| + CatBoost equal-weight | 0.1610 | Ensemble diversity | |
| + Temperature | 0.1608 | Mild calibration improvement | |
| Remove h2h (19 features) | 0.1619 | Fixes train/inference distribution shift | CV +0.0011, but h2h=0 in Stage 2; generalization improves |
| **Final (3-seed LGB + CB)** | **0.1619** | Multi-seed for stability | Submitted version |

**Improvement over seed-only**: 0.1770 -> 0.1619 = **8.5% reduction**
**Improvement over naive**: 0.2500 -> 0.1619 = **35.2% reduction**

---

## Tournament Performance (Ongoing)

### Day 1 Results (16 men's R64 games)

- Public LB: 0.15242
- Correct favorites: 10/16
- 6 upsets killed our Brier (contributed 99% of total error)
- Non-upset average Brier: 0.0034 (near-perfect)
- Worst predictions: High Point > Wisconsin (Brier=0.62), Texas A&M > St Mary's (0.45)

### Women's Games Advantage

- Women's model (Brier=0.137) much stronger than men's (0.184)
- As women's games enter scoring, our overall score should improve
- Duke W and TCU W games: estimated to pull score from 0.152 to ~0.136

### Medal Chances (estimated after Day 1)

- Bronze (~0.141): 15-20%
- Silver (~0.135): 10-15%
- Gold (~0.120): < 5%

---

## Lessons Learned

1. **Less is more**: 19 features beat 171 features. 2 models beat 6 models. 6 free parameters beat 15+.
2. **Train/inference distribution matters more than CV score**: Removing h2h made CV worse but was objectively the right call.
3. **Free parameter budget**: With only 795 validation samples, every optimized parameter has a cost. The 132:1 ratio was achieved through aggressive simplification.
4. **Ensemble diversity needs quality**: Low correlation (0.843) isn't enough if the model is bad (Brier 0.223).
5. **Gender-specific modeling is essential**: Women's tournament is far more predictable; a unified model leaves performance on the table.
6. **Aggressive predictions favor Brier Score**: T < 1.0 is mathematically optimal when your model is well-calibrated.
7. **Probability calibration on upsets drives Brier variance**: 6 upsets (38% of games) contributed 99% of Day 1 error. But the issue is not that upsets happened -- it's that our model assigned them too-low probabilities (e.g., 21% for High Point, 37% for VCU). A perfectly calibrated model would assign ~30-40% to plausible upsets, not 3%. Brier Score measures probability calibration quality, not prediction accuracy.

---

## Team Collaboration

This was a team project. Key teammate contributions:
- **Feature selection validation**: Teammate independently ran the pipeline with best_k=20 and got Brier 0.1568 (vs our 0.1599 with k=50). This was the critical evidence that fewer features generalize better, and directly led to fixing k=20.
- **Optuna seed sensitivity discovery**: Running the same pipeline with different random seeds produced 0.001-0.003 Brier variation, making small "improvements" indistinguishable from noise. This motivated fixing seed=42 and adding the 0.003 reliability threshold.

---

## What We Would Do Differently

1. **Start simple, stay simple**: We built a 6-model pipeline with 15+ free parameters before realizing it was overfitting. If starting over, we would begin with LGB + top-20 features + equal-weight CB from Day 1, and only add complexity with strong evidence (sign test 6/6 folds).
2. **Check train/inference distribution shift immediately**: The h2h feature was in the model for weeks before we noticed it's always zero in Stage 2. A simple check (`stage2_df[feature].nunique()`) on Day 1 would have caught this instantly.
3. **Invest more in upset calibration**: Our biggest Day 1 errors were all upsets where we gave < 25% probability. Exploring features that capture upset potential (e.g., pace mismatch, experience vs athleticism) could have improved calibration in the 20-40% range.
4. **Test submission format earlier**: We had 3 failed submissions (format errors) before the first successful one. Validating against the sample submission CSV should have been automated from the start.

---

## Interview-Ready Stories (5-minute narratives)

1. **The h2h Discovery**: How a top-5 importance feature turned out to be completely useless in production, and why removing it made CV worse but the model better. Demonstrates understanding of distribution shift and the difference between in-sample importance and out-of-sample value.

2. **The Simplification Arc**: How we went from 6 models / 15 parameters to 2 models / 6 parameters and got a better result. The "free parameter budget" framework and why 795 samples can't support 15 degrees of freedom.

3. **The Spread Model Rejection**: Why we rejected a model with 0.843 correlation (below our 0.90 threshold for inclusion) because its absolute Brier of 0.2229 was too poor. Demonstrates that ensemble theory (low correlation = good) has practical limits.

4. **The Temperature Direction Debate**: Mathematical proof that T < 1.0 (aggressive) beats T > 1.0 (conservative) for Brier Score, with confirmation from the gold solution. Shows ability to reason from first principles about loss functions.
