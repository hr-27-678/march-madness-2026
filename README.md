# 🏀 NCAA March Madness 2026 — Kaggle Competition

Predict win probabilities for all possible matchups in the 2026 NCAA Basketball Tournament (Men & Women). Optimized for **Brier Score** (lower is better).

**Kaggle Competition**: [March Machine Learning Mania 2026](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)

## Results

| Metric | Score |
|--------|-------|
| CV Brier (6-fold time-series) | **0.1619** |
| Public Leaderboard (Day 1) | **0.0960** (34 games evaluated) |
| vs Seed-only baseline (0.177) | -8.5% |
| vs Naive baseline (0.250) | -35.2% |

## Architecture

```
35 CSV files (2003-2025 seasons)
    │
    ▼
Feature Pipeline ── 156 team-level features → 171 matchup diff features
    │
    ▼
Feature Selection ── Top-19 by LGB importance (stable across all 6 CV folds)
    │
    ▼
Gender Split ── Men (1,449 games) │ Women (1,402 games)
    │
    ▼
Models ── LightGBM × 3 seeds + CatBoost (depth=4)
    │        Men: CatBoost uses augmented data (tourney + regular season)
    │        Women: tournament-only
    ▼
Equal-Weight Ensemble ── 0.5 × LGB + 0.5 × CB (zero free parameters)
    │
    ▼
Temperature Scaling ── T=0.90 (pushes confident predictions further)
    │
    ▼
132,133 matchup predictions
```

**Total optimized parameters**: 5 (Optuna) + 0 (weights) + 1 (temperature) = **6 params / 795 samples = 1:132 ratio**

## Project Structure

```
├── README.md
├── PROJECT_SUMMARY.md          # Detailed technical writeup
├── notebooks/
│   └── ncaa_march_madness_2026.ipynb   # Main pipeline notebook
├── src/
│   ├── pipeline.py             # End-to-end feature pipeline
│   ├── features.py             # 30 basic + 63 advanced basketball stats
│   ├── elo.py                  # 3 Elo variants (standard/aggressive/conservative)
│   ├── massey.py               # Massey Ordinals aggregation (men only)
│   ├── context_features.py     # Conference, coach, seed features
│   ├── data_loader.py          # Data loading utilities
│   └── validate.py             # CV and evaluation framework
└── submissions/
    ├── stage1/                 # Backtest predictions (2022-2025)
    └── stage2/                 # Final tournament predictions
```

## Key Technical Decisions

### 1. Feature Selection: 171 → 19 features
LGB with all 171 features (Brier=0.170) was **worse** than 7-feature Logistic Regression (0.166). Aggressive feature selection to top-19 by importance, validated stable across all 6 CV folds.

### 2. Removing h2h Features (Most Impactful Decision)
`h2h_avg_margin` ranked #5 by importance but was **always zero in Stage 2** (future games have no head-to-head history). This is a textbook train/inference distribution shift. Removing it made CV slightly worse (+0.001) but fixed generalization.

### 3. Simplification Over Complexity
Started with 6 models and 15+ free parameters. Diagnosed overfitting (MLP weight=0.42, Spread weight=0.75 — artifacts of optimizing on 795 samples). Simplified to 2 models, 6 parameters, and got better results.

### 4. Spread Model Rejection
Implemented XGB spread regression → Isotonic calibration. OOF correlation with LGB was 0.843 (low enough to add diversity). But absolute Brier of 0.223 was worse than the seed-only baseline — **low correlation is necessary but not sufficient**.

### 5. Temperature Direction (T < 1.0)
Mathematical analysis: for Brier Score, aggressive predictions (T < 1.0) outperform conservative ones when the model is well-calibrated. A 97% prediction correct: Brier=0.0009 vs 80% prediction correct: Brier=0.04 — 44× difference. Confirmed by [paris-madness-2023 gold solution](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/discussion/405253) approach.

## Validation

6-fold time-series CV (train on seasons < N, validate on season N's tournament):

| Fold | Season | Games | Brier |
|------|--------|-------|-------|
| 0 | 2019 | 130 | varies by model |
| 1 | 2021 | 129 | (2020 skipped — COVID) |
| 2 | 2022 | 134 | |
| 3 | 2023 | 134 | |
| 4 | 2024 | 134 | |
| 5 | 2025 | 134 | |
| **Total** | | **795** | **0.1619** |

## Model Progression

| Version | CV Brier | Change |
|---------|----------|--------|
| Seed-only baseline | 0.1770 | — |
| Logistic Regression (7 features) | 0.1661 | Simple but strong |
| LGB (top-20) + Optuna | 0.1637 | Feature selection + tuning |
| + CatBoost equal-weight | 0.1610 | Ensemble diversity |
| + Remove h2h (19 features) | 0.1619 | CV worse, generalization better |
| + Multi-seed LGB (×3) | **0.1619** | Stability, final version |

## Tournament Performance

*Tournament is still in progress — results will be updated after completion.*

## Lessons Learned

1. **Less is more** — 19 features > 171 features. 2 models > 6 models. 6 free parameters > 15+.
2. **Train/inference distribution shift > CV score** — Removing h2h made CV worse but was the right call.
3. **Free parameter budget** — 795 validation samples can't support 15 optimized parameters. The 1:132 ratio was achieved through aggressive simplification.
4. **Ensemble diversity needs quality** — Low correlation isn't enough if the model is bad.
5. **Gender-specific modeling is essential** — Women's Brier (0.137) far outperforms men's (0.184).
6. **Probability calibration on upsets drives variance** — 6 upsets contributed 99% of Day 1 error. Brier Score measures calibration quality, not just prediction accuracy.

## Tech Stack

- Python, LightGBM, CatBoost, XGBoost, Optuna
- scikit-learn (Isotonic Regression, calibration)
- pandas, numpy, scipy
- Jupyter Notebook

## Team

This is a team competition. I collaborated with Xinwei Huang on feature engineering and experiment design. This repo contains my implementation.

## Detailed Writeup

See [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for the full technical deep-dive, including:
- Complete feature engineering details
- Overfitting diagnosis methodology
- Past winning solution analysis
- Interview-ready technical narratives
