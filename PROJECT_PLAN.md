# NCAA March Madness 2026 - Kaggle Competition Project Plan

## Competition Overview
- **Goal**: Predict P(TeamA beats TeamB) for every possible team matchup (men's + women's)
- **Metric**: Brier Score (= MSE of predicted probabilities vs actual 0/1 outcomes)
- **Submission**: `ID,Pred` where ID = `2026_XXXX_YYYY`, Pred = P(lower TeamID wins)
- **Stage1**: Backtest on 2022-2025 seasons (519K rows); **Stage2**: Predict 2026 (132K rows)

## Data Inventory Summary

| Category | Key Files | Rows | Seasons |
|----------|-----------|------|---------|
| Men's Regular Season (Detailed) | MRegularSeasonDetailedResults | 124K | 2003-2026 |
| Men's Regular Season (Compact) | MRegularSeasonCompactResults | 198K | 1985-2026 |
| Men's Tournament (Detailed) | MNCAATourneyDetailedResults | 1,450 | 2003-2025 |
| Men's Tournament (Compact) | MNCAATourneyCompactResults | 2,586 | 1985-2025 |
| Women's Regular Season (Detailed) | WRegularSeasonDetailedResults | 86K | 2010-2026 |
| Women's Tournament (Detailed) | WNCAATourneyDetailedResults | 962 | 2010-2025 |
| Rankings (Massey Ordinals) | MMasseyOrdinals | 5.8M | 2003-2026 (men only) |
| Seeds | MNCAATourneySeeds / WNCAATourneySeeds | 2.6K / 1.7K | 1985 / 1998 |
| Coaches | MTeamCoaches | 13.9K | 1985-2026 |
| Conferences | MTeamConferences / WTeamConferences | 13.7K / 9.8K | 1985 / 1998 |
| Game Locations | MGameCities / WGameCities | 91K / 88K | 2010-2026 |

---

## Phase 0: Data Preprocessing & Infrastructure

### 0.1 统一男女数据结构
- 男女数据结构完全一致，男队ID 1000-1999，女队ID 3000-3999，不重叠
- 合并处理：给数据加 `gender` 列，或直接 concat（ID不重叠天然分离）
- 注意：女队没有 Massey Ordinals 数据，需要单独处理

### 0.2 构建"对称化"的训练数据
- 原始数据以 Winner/Loser 组织，需要转换为 **TeamA vs TeamB** 格式
- 每场比赛生成两行：(TeamA, TeamB, label=1) 和 (TeamB, TeamA, label=0)
- 或者：统一用 lower ID 作为 TeamA，higher ID 作为 TeamB（与submission格式一致）
- **label**: 1 = lower ID team wins, 0 = higher ID team wins

### 0.3 时间窗口定义
- DayNum 0-132: Regular season + Conference tournaments
- DayNum 133: Final rankings available
- DayNum 134+: NCAA Tournament
- 训练集：使用当年 regular season 数据预测当年 tournament 结果

---

## Phase 1: Feature Engineering (6大类，~100+特征)

### 1.1 基础赛季统计特征 (Team-Season Level)
**方法：聚合每支队伍整个赛季的表现**

对每支队伍每个赛季计算：
```
- 胜率 (win_rate), 主场胜率, 客场胜率, 中立场胜率
- 场均得分 (avg_score), 场均失分 (avg_opp_score), 场均分差 (avg_margin)
- 得分标准差 (score_std) — 衡量稳定性
- 加权胜率（按对手强度加权，SOS-adjusted）
- 近期表现：最后30天/20场的胜率和分差（momentum）
- Conference tournament 表现：是否赢得conference tournament
```

### 1.2 高级篮球统计特征 (Advanced Box Score)
**方法：从 Detailed Results 计算 Four Factors + 高级指标**

```python
# Four Factors (Dean Oliver) — 最重要的篮球分析指标
- eFG% = (FGM + 0.5 * FGM3) / FGA           # 有效投篮命中率
- TO_rate = TO / (FGA + 0.44 * FTA + TO)      # 失误率 (possessions-based)
- OR% = OR / (OR + Opp_DR)                    # 进攻篮板率
- FT_rate = FTM / FGA                         # 罚球率

# Tempo & Efficiency
- possessions ≈ FGA - OR + TO + 0.44 * FTA   # 回合数估算
- offensive_efficiency = Score / possessions * 100
- defensive_efficiency = Opp_Score / possessions * 100
- net_efficiency = off_eff - def_eff          # 净效率（最强单一预测指标之一）

# 其他
- assist_rate = Ast / FGM
- block_rate = Blk / Opp_FGA
- steal_rate = Stl / Opp_possessions
- 3pt_rate = FGA3 / FGA                       # 三分球依赖度
- FT_accuracy = FTM / FTA
- rebound_margin = (OR + DR) - (Opp_OR + Opp_DR)
```

**每个指标计算：mean, median, std, trend (线性回归斜率)**

### 1.3 排名系统特征 (Massey Ordinals)
**方法：利用 5.8M 排名数据提取多维度排名特征（仅男队）**

```
- 各系统最终排名 (RankingDayNum=133): 选择 top-N 个系统
- 重点系统: POM (KenPom), SAG (Sagarin), MOR (Morley), RPI, BPI, AP, USA
- 排名聚合: mean_rank, median_rank, min_rank, max_rank, std_rank
- 排名趋势: 赛季末排名 vs 赛季中排名的变化（趋势）
- 系统间一致性: rank_std 低 → 多个系统一致看好
- 排名百分位: rank / total_teams
```

**女队无 Massey 数据的处理方案:**
1. 用自建的 Elo/效率指标替代
2. 训练时男女分别建模，或用 flag 区分

### 1.4 Elo Rating 系统特征 (自建)
**方法：实现自定义 Elo 评分系统**

```python
# 基础 Elo
- K-factor: 32 (可调参)
- 主场优势: +100 (可调参)
- 每赛季初始化: 0.75 * last_season_elo + 0.25 * 1500 (均值回归)

# 增强版 Elo
- 考虑胜分差的 Elo (margin-based):
  MOV_multiplier = log(abs(margin) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
- 分开计算进攻/防守 Elo
- 最近N场的 Elo 变化率（动量指标）

# 女队也可计算 Elo → 弥补无 Massey 的缺陷
```

### 1.5 对战特征 (Matchup-Level)
**方法：构建 TeamA vs TeamB 的交互特征**

```
- 特征差值: TeamA_feature - TeamB_feature (所有team-level特征)
- 特征比值: TeamA_feature / TeamB_feature
- 种子差: seed_A - seed_B (仅tournament有)
- 种子交叉: seed_A * seed_B
- 历史交锋: 过去5年直接对战胜率（如果有）
- Conference 关系: same_conference flag, conference_power_rank差
- Elo差值: elo_A - elo_B
- 排名差值: rank_A - rank_B
```

### 1.6 上下文特征 (Contextual)
**方法：加入非技术面的结构化信息**

```
- 种子 (seed number): 1-16, 转为数值
- 种子区域 (region): W/X/Y/Z → 可能影响对阵路径
- 教练经验: coach_seasons (该教练总执教年数)
- 教练 tournament 经验: 过去参加NCAA tournament的次数
- Conference 强度: conference内所有队伍的平均排名
- Conference tournament 冠军: binary flag
- 主客场旅行距离 (可选): 用城市数据算距离
```

### 1.7 特征选择策略

**尝试1: 全特征 baseline**
- 所有特征直接输入模型，让模型自行选择

**尝试2: 基于重要性的筛选**
- 先跑一轮 LightGBM, 按 feature importance 保留 top-50
- 或用 Boruta / 递归特征消除 (RFE)

**尝试3: 相关性过滤**
- 去除相关性 > 0.95 的冗余特征
- 保留与 target 相关性最高的那个

**尝试4: PCA/特征降维**
- 对高度相关的 box score 统计做 PCA
- 保留 95% 方差的主成分

---

## Phase 2: Model Selection & Training

### 2.1 Baseline Models

**Model 1: Logistic Regression (LR)**
- 最简单的概率输出模型
- 用 seed 差和 Elo 差作为基础特征
- 作为所有其他模型的 benchmark
- 预期 Brier Score 基准

**Model 2: LightGBM (核心模型)**
- 最适合表格数据的 GBDT 模型
- 优势：自动处理缺失值、类别特征；训练快
- 直接输出 `objective='binary'` 的概率
- **推荐作为主力模型**

**Model 3: XGBoost**
- 与 LightGBM 互补
- 不同的分裂策略和正则化
- ensemble 时提供多样性

**Model 4: CatBoost**
- 对类别特征有原生支持
- 不同的 boosting 策略 (Ordered Boosting)
- 减少 target leakage

### 2.2 神经网络方案

**Model 5: TabNet / Neural Network**
- 自动特征选择的注意力机制
- 可以捕获复杂的非线性交互
- 需要更多调参

**Model 6: Team Embedding + Neural Network**
- 给每支队伍学习一个 embedding vector (如 dim=32)
- 输入：concat(TeamA_embedding, TeamB_embedding, matchup_features)
- 用全连接网络输出胜率
- **优势：能学到数据中未被显式特征捕获的队伍"风格"**
- 可考虑使用 historical game sequences 做 embedding pre-training

### 2.3 概率校准模型

**Model 7: Bradley-Terry Model**
- 经典的配对比较模型
- P(A beats B) = strength_A / (strength_A + strength_B)
- 可以 MLE 估计每队 strength parameter
- 简单但在 bracket prediction 中历史表现不错

### 2.4 Ensemble / Stacking 策略

**是否需要 ensemble？—— 强烈建议，这是 Kaggle 竞赛的制胜关键**

**方案A: Simple Weighted Average**
```python
pred = w1 * lgb_pred + w2 * xgb_pred + w3 * cat_pred + w4 * lr_pred
# 权重通过 CV 上的 Brier Score 优化 (scipy.optimize)
```

**方案B: Stacking (两层)**
```
Layer 1 (Base Models):
  - LightGBM (多组超参, 如 3-5 个不同配置)
  - XGBoost
  - CatBoost
  - Logistic Regression
  - Neural Network / TabNet

Layer 2 (Meta Model):
  - Logistic Regression (推荐, 防止过拟合)
  - 或 Ridge Regression
  - 输入: Layer 1 各模型的 OOF predictions
  - 输出: 最终概率
```

**方案C: Blending**
- 用 holdout set 而非 CV fold 训练 meta model
- 更简单，但数据利用率低一些

**推荐路线**: 先做好单模型 (LightGBM)，确认 pipeline 正确，再逐步加入 ensemble。

---

## Phase 3: Cross-Validation 策略 (极其关键)

### 3.1 为什么 CV 策略至关重要？
- NCAA 数据有强时序性：不能用未来数据预测过去
- Tournament 样本极少（每年仅 ~63 场男子 + ~63 场女子）
- CV 分数直接决定模型选择和超参方向

### 3.2 CV 方案对比

**方案1: 时序 CV (推荐作为主方案)**
```
Fold 1: Train on 2003-2018, Validate on 2019 tournament
Fold 2: Train on 2003-2019, Validate on 2021 tournament (跳过2020)
Fold 3: Train on 2003-2021, Validate on 2022 tournament
Fold 4: Train on 2003-2022, Validate on 2023 tournament
Fold 5: Train on 2003-2023, Validate on 2024 tournament
Fold 6: Train on 2003-2024, Validate on 2025 tournament
```
- 完全模拟真实预测场景
- **缺点**: 每fold验证集很小(~63场), 方差大

**方案2: 扩展时序 CV (推荐配合方案1)**
- 训练集: 用 regular season + tournament 结果
- 验证集: 仅用 tournament 结果
- 可以加入 regular season 作为辅助验证（但权重降低）

**方案3: Leave-One-Season-Out (LOSO)**
- 每次留出一整个赛季的 tournament 作为验证
- 用其余所有赛季训练
- 获得更多fold, 但打破了时序

**方案4: Grouped K-Fold (按赛季分组)**
```
- 5-fold, 每 fold 包含几个赛季的 tournament
- 保证同一赛季的比赛不会同时出现在 train 和 val
- 比 LOSO 更粗粒度
```

**推荐策略**:
- **主 CV**: 时序 CV (方案1), 用于最终模型选择和超参调优
- **辅助 CV**: LOSO (方案3), 用于特征工程快速迭代
- **Stage1 验证**: 2022-2025 的 tournament 是已知结果，可以作为最终检验

### 3.3 训练数据策略

**用什么数据训练？有3种选择：**

**选择A: 仅用 Tournament 数据训练**
- 优点：分布最接近预测目标
- 缺点：数据量太少 (~1450 场男 + ~960 场女 detailed)

**选择B: Regular Season + Tournament 混合训练**
- 优点：数据量大
- 缺点：regular season 和 tournament 分布不同
- 可以给 tournament 样本更高的 sample_weight

**选择C: 仅用 Regular Season 数据做特征，用 Tournament 数据训练 (推荐)**
- Features: 从 regular season 统计中计算
- Labels: 来自 tournament 结果
- 这是最合理的方案：用已知信息预测未知比赛

---

## Phase 4: Data Augmentation & 数据问题处理

### 4.1 数据增强方法

**方法1: 对称翻转 (最基础且必要)**
- 每场 (A beats B) → 同时生成 (B loses to A)
- 保证模型学到的关系是对称的
- 注意翻转时 label 和特征差值都要取反

**方法2: 历史窗口滑动**
- 不只用赛季末的统计，也用赛季中间时间点的统计
- 例如: 截至 DayNum=100 的统计, DayNum=110 的统计, DayNum=120 的统计
- 每个时间点生成一个样本 → 数据量 x3

**方法3: Bootstrap / Noise Injection**
- 对特征加入小随机噪声 (±2-5%)
- 可以作为训练时的 regularization
- 在 GBDT 中效果有限, 在神经网络中更有用

**方法4: 跨性别迁移学习**
- 男女篮球的基本规律类似
- 可以用男队数据预训练, 然后 fine-tune 到女队
- 或者直接合并训练 (加 gender flag)

**方法5: Regular Season 对战数据作为额外训练**
- 把 regular season 的对战结果也加入训练
- 但给 tournament 样本 2-5x 的 sample_weight
- 这极大增加了训练数据量

### 4.2 过拟合 vs 欠拟合处理

**过拟合风险 (主要问题):**
- Tournament 数据极少 (~2400 场有 detailed stats)
- 特征数量可能远超有效样本数
- **对策:**
  1. 强正则化: LightGBM 的 `min_child_samples=50+`, `reg_alpha`, `reg_lambda`
  2. 限制树深度: `max_depth=4-6`
  3. 特征选择: 先大胆加特征, 再严格筛选
  4. Early stopping: 基于 CV 的 Brier Score
  5. Stacking 时 meta model 用 L2 正则化的 LR

**欠拟合检查:**
- 如果 train Brier Score 和 val Brier Score 都很高 → 欠拟合
- **对策:** 加更多特征, 增加模型复杂度, 检查特征工程

### 4.3 Embedding 方法 (高级数据增强)

**Team2Vec: 类似 Word2Vec 的思路**
```
- 把每场比赛看作一个 "sentence": [TeamA, TeamB, score_diff, ...]
- 用 skip-gram 或 CBOW 学习 team embedding
- 输入所有历史比赛, 输出每队的 dense vector (dim=16-64)
- 能捕捉到 "playing style similarity"
```

**Game Sequence Embedding:**
```
- 每支队伍的赛季可以看作一个 sequence of games
- 用 LSTM/Transformer 编码这个序列
- 输出: 队伍当前状态的 embedding
- 能捕捉动量、连胜/连败等时序模式
```

**Graph Embedding:**
```
- 构建赛季内的比赛图: 节点=队伍, 边=比赛结果
- 用 Node2Vec / GraphSAGE 学习 embedding
- 能捕捉到间接对战关系 (A > B > C → A 可能 > C)
```

---

## Phase 5: 高级优化技巧

### 5.1 Probability Calibration (概率校准)
**Brier Score 对概率校准极其敏感!**

```python
# 方法1: Platt Scaling
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)

# 方法2: Isotonic Regression
calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)

# 方法3: Temperature Scaling (推荐)
# 学习一个 temperature T, 使得 calibrated_prob = sigmoid(logit(raw_prob) / T)

# 方法4: Beta Calibration
# 更灵活的校准方法, 适合体育预测
```

**校准检查工具:**
- Reliability diagram (校准曲线)
- Expected Calibration Error (ECE)

### 5.2 Seed-Based Prior / Post-Processing
```
历史数据中种子差 vs 胜率的统计:
- 1 vs 16: ~99% 胜率
- 2 vs 15: ~94%
- 3 vs 14: ~85%
- ...
- 8 vs 9: ~52%

# 可以用种子历史胜率作为 prior
# 最终预测 = alpha * model_pred + (1-alpha) * seed_prior
# alpha 通过 CV 调优 (通常 0.7-0.9)
```

### 5.3 Upset Probability Tuning
- NCAA tournament 以"黑马"(upsets)闻名
- 模型可能对强队过度自信
- **检查**: 如果预测概率 > 0.95 或 < 0.05, 进行 clipping
- **推荐**: clip predictions to [0.02, 0.98] 或更保守的 [0.05, 0.95]
- 具体 clip 范围通过 CV 调优

### 5.4 Gender-Specific Modeling
```
方案A: 完全分开建模
  - 男女各自一套 pipeline
  - 男队有 Massey data 优势
  - 女队需要自建 ranking 替代

方案B: 统一模型 + gender flag
  - 合并训练, 加 is_mens 特征
  - 简单, 数据量更大

方案C: 混合方案 (推荐)
  - 共享基础特征工程 pipeline
  - 分别训练模型
  - ensemble 时可以有不同的权重
```

### 5.5 Hyperparameter Tuning

**LightGBM 关键超参:**
```python
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',  # 或直接用 brier score 自定义
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [15, 31, 63],
    'max_depth': [4, 5, 6, 7],
    'min_child_samples': [20, 50, 100],
    'feature_fraction': [0.6, 0.8, 1.0],
    'bagging_fraction': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
    'n_estimators': 1000-5000 (with early stopping)
}
```

**调参策略:**
1. **Optuna** (推荐): Bayesian optimization, 自动搜索
2. 先粗搜 (大范围 random search), 再细搜 (小范围 grid search)
3. **始终用时序 CV** 评估, 不要用 random split

### 5.6 Loss Function 优化
```
- 默认: binary cross-entropy (logloss)
- 可尝试: 直接优化 Brier Score 作为自定义 loss
  - Brier = mean((pred - actual)^2)
  - 梯度和 logloss 不同, 可能给出不同的概率分布
- Focal Loss: 更关注难分类样本 (upset games)
```

### 5.7 外部数据 (谨慎使用)
```
可考虑的额外数据源:
- KenPom ratings (kenpom.com) — 但可能要付费
- ESPN BPI
- 球队伤病信息 (难以标准化)
- 赌博市场赔率 (非常有信息量, 但获取需谨慎)
```

---

## Phase 6: Pipeline & Execution Plan

### Step 1: 基础 Pipeline (Week 1)
```
1. 数据加载 & 清洗
2. 基础特征: seed, win_rate, avg_margin, Elo
3. LightGBM baseline + 时序CV
4. 生成 Stage1 submission 验证流程正确性
```

### Step 2: 特征工程迭代 (Week 2)
```
5. 添加 Four Factors / Advanced Stats
6. 添加 Massey Ordinals 特征 (男队)
7. 自建 Elo 系统 (男女通用)
8. 教练/conference 特征
9. 每加一批特征, 用 CV 验证是否有提升
```

### Step 3: 模型调优 (Week 3)
```
10. LightGBM 超参优化 (Optuna, 100+ trials)
11. 训练 XGBoost, CatBoost
12. 尝试 Neural Network / Team Embedding
13. Probability Calibration
```

### Step 4: Ensemble & 最终优化 (Week 4)
```
14. Stacking / Weighted Average
15. Upset clipping / Seed prior blending
16. 最终 CV 评估
17. 生成 Stage2 submission
```

---

## Key Insights & 注意事项

1. **Brier Score 特性**: 对极端概率的错误惩罚极大。预测 0.95 但实际输了 → penalty = 0.9025。所以**概率校准和 clipping 比模型选择更重要**。

2. **种子是最强特征**: 历史上仅用 seed 差就能达到不错的 Brier Score。任何模型都应包含 seed 信息。

3. **数据泄漏防范**:
   - 永远不要用 tournament 期间的数据去预测同一 tournament
   - 确保 Massey Ordinals 的 RankingDayNum ≤ 133
   - Elo 系统只用到 regular season 结束的值

4. **女队挑战**: 无 Massey 排名，detailed stats 从 2010 才开始。需要更多自建指标。

5. **2020赛季缺失**: COVID 导致 2020 tournament 取消。训练时需要跳过。

6. **Stage1 是免费验证集**: 2022-2025 的 tournament 结果可以直接评估模型。

7. **提交策略**: 可以多次提交，但**一定要手动选择最终提交**，不要依赖自动选择。

---

## Expected Brier Score Targets
| Model | Expected Brier Score |
|-------|---------------------|
| All 0.5 (naive) | ~0.250 |
| Seed-only logistic | ~0.200 |
| Good single model | ~0.170-0.180 |
| Top ensemble | ~0.155-0.165 |
| Competition winner level | ~0.145-0.155 |

---

## File Structure Plan
```
NCAA marchmadness2026/
├── data/                          # 原始数据 (已有)
├── notebooks/
│   ├── 01_EDA.ipynb              # 探索性分据分析
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_ensemble.ipynb
├── src/
│   ├── data_loader.py            # 数据加载
│   ├── features.py               # 特征工程
│   ├── elo.py                    # Elo系统
│   ├── models.py                 # 模型训练
│   ├── cv.py                     # 交叉验证
│   ├── calibration.py            # 概率校准
│   └── submission.py             # 生成提交文件
├── submissions/
│   ├── stage1/
│   └── stage2/
└── PROJECT_PLAN.md               # 本文件
```
