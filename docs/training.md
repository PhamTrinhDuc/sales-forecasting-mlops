# Model Training Documentation

## 📖 Tổng quan

Hệ thống sử dụng 2 mô hình Gradient Boosting để dự đoán doanh số:
- **XGBoost** (eXtreme Gradient Boosting)
- **LightGBM** (Light Gradient Boosting Machine)

Cả hai đều thuộc họ **ensemble learning** - kết hợp nhiều weak learners (decision trees) thành một strong learner.

---

## 🌳 Gradient Boosting là gì?

### Ý tưởng cốt lõi

Thay vì train 1 model phức tạp, gradient boosting train **nhiều models đơn giản tuần tự**, mỗi model học từ lỗi của model trước:

```
Step 1: Train Tree₁ → Prediction₁ → Error₁
Step 2: Train Tree₂ để predict Error₁ → Prediction₂ → Error₂
Step 3: Train Tree₃ để predict Error₂ → Prediction₃ → Error₃
...
Step N: Final Prediction = Tree₁ + Tree₂ + Tree₃ + ... + Treeₙ
```

### Ví dụ cụ thể

**Mục tiêu:** Dự đoán sales = 1000

```chạy
Round 1: Tree₁ predict 800   → Error = 200
Round 2: Tree₂ predict 150   → Error = 50
Round 3: Tree₃ predict 40    → Error = 10
Round 4: Tree₄ predict 8     → Error = 2
...
Final:   800+150+40+8 = 998  → Very close to 1000!
```

Mỗi tree chỉ cần học **phần còn thiếu** (residual), không phải toàn bộ pattern.

### Learning Rate (Shrinkage)

```python
learning_rate = 0.1  # Shrinkage factor
Final = learning_rate × (Tree₁ + Tree₂ + ... + Treeₙ)
```

**Tại sao cần learning rate?**
- Learning rate cao (0.3): Learn nhanh, dễ overfit
- Learning rate thấp (0.01): Learn chậm, stable hơn, cần nhiều trees hơn

**Trade-off:**
- `learning_rate = 0.3`, `n_estimators = 100` → Fast but risky
- `learning_rate = 0.01`, `n_estimators = 1000` → Slow but robust

---

## 🔥 XGBoost Deep Dive

### Cách hoạt động

**1. Level-wise Tree Growth**

XGBoost grow trees theo **level** (tầng):

```
         [Root]           ← Level 0
        /      \
      [A]      [B]        ← Level 1 (grow cả 2 nodes)
     /  \     /  \
   [C] [D] [E]  [F]       ← Level 2 (grow cả 4 nodes)
```

- Grow tất cả nodes cùng level trước khi xuống level tiếp theo
- **Ưu điểm:** Balanced tree, tránh quá sâu
- **Nhược điểm:** Không tận dụng hết potential của từng branch

**2. Training Process**

```python
params = {
    "n_estimators": 200,       # Max 200 trees
    "max_depth": 6,            # Max depth per tree
    "learning_rate": 0.1,      # Shrinkage
    "subsample": 0.8,          # 80% samples per tree
    "colsample_bytree": 0.8,   # 80% features per tree
}
```

**Mỗi round (tree):**

```
1. Sample 80% of training data (subsample=0.8)
2. Sample 80% of features (colsample_bytree=0.8)
3. Build tree với max_depth=6
4. Calculate gradients (errors) từ previous predictions
5. Fit tree để predict gradients
6. Update predictions: pred_new = pred_old + learning_rate × tree_pred
7. Evaluate trên validation set
```

**3. Regularization**

XGBoost có nhiều cơ chế regularization để tránh overfitting:

```python
"gamma": 0.1,          # Min loss reduction để split node
"reg_alpha": 0.5,      # L1 regularization on weights
"reg_lambda": 1.0,     # L2 regularization on weights
"min_child_weight": 3  # Min sum of weights in child
```

**Gamma:**
- Node chỉ split nếu loss reduction > gamma
- Gamma cao → Ít split → Tree đơn giản hơn

**L1/L2 Regularization:**
- Penalty trên leaf weights
- Giảm weights → Predictions mượt hơn → Less overfitting

### Hyperparameters Quan trọng

| Parameter | Range | Ảnh hưởng |
|-----------|-------|-----------|
| `n_estimators` | 50-300 | Số trees. Nhiều = mạnh hơn nhưng chậm + risk overfit |
| `max_depth` | 3-10 | Độ sâu tree. Sâu = capture complex patterns |
| `learning_rate` | 0.01-0.3 | Tốc độ học. Thấp = stable, cần nhiều trees |
| `subsample` | 0.6-1.0 | % samples/tree. <1 = stochastic, reduce overfit |
| `colsample_bytree` | 0.6-1.0 | % features/tree. <1 = diversity giữa trees |
| `gamma` | 0-0.5 | Min loss để split. Cao = conservative |
| `reg_alpha` | 0-1.0 | L1 regularization. Cao = sparse weights |
| `reg_lambda` | 0-1.0 | L2 regularization. Cao = smooth weights |

### Early Stopping

```python
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50,
    verbose=True
)
```

**Cơ chế:**

1. Sau mỗi round, tính validation metric (RMSE, MAE, etc.)
2. Track best metric và best iteration
3. Nếu **50 rounds liên tiếp** không cải thiện → STOP
4. Return model với weights của **best iteration**

**Timeline thực tế:**

```
Round   Val_RMSE   Best_RMSE   Patience_Counter
------  ---------  ----------  ----------------
1       450.2      450.2       0
20      420.5      420.5       0
50      398.3      398.3       0
100     385.7      385.7       0
142     382.1      382.1       0  ← BEST!
143     382.3      382.1       1
144     382.8      382.1       2
...
192     383.5      382.1       50 → STOP!

→ Model sử dụng weights của Round 142
```

**Tại sao cần early stopping?**
- ✅ Tránh overfit (train quá lâu)
- ✅ Tiết kiệm thời gian
- ✅ Tự động tìm optimal số trees

---

## 💡 LightGBM Deep Dive

### Khác biệt với XGBoost

**1. Leaf-wise Tree Growth**

LightGBM grow theo **leaf** (lá), không phải level:

```
XGBoost (level-wise):        LightGBM (leaf-wise):
         [Root]                      [Root]
        /      \                    /      \
      [A]      [B]                [A]      [B]
     /  \     /  \                          \
   [C] [D] [E]  [F]                         [C]
                                              \
                                              [D]
```

- Chọn leaf có **highest loss reduction** để split
- **Ưu điểm:** Hiệu quả hơn, accuracy cao hơn với ít trees hơn
- **Nhược điểm:** Dễ overfit nếu không regularize

**2. Histogram-based Learning**

Thay vì xét tất cả split points, LightGBM:
- Chia features thành **bins** (histogram)
- Chỉ xét split tại bin boundaries
- → Nhanh hơn nhiều, đặc biệt với large datasets

**3. Gradient-based One-Side Sampling (GOSS)**

- Giữ lại samples có **large gradients** (learn nhiều)
- Random sample một phần samples có **small gradients**
- → Giảm computation mà không mất nhiều information

### Hyperparameters Đặc biệt

```python
params = {
    "num_leaves": 31,          # Max số lá (không phải depth!)
    "min_child_samples": 20,   # Min samples trong 1 leaf
    "max_bin": 255,            # Số bins cho histogram
    "boosting_type": "gbdt",   # Gradient Boosting Decision Tree
}
```

**num_leaves:**
- Quan trọng nhất cho LightGBM
- `num_leaves = 2^max_depth` (nếu balanced tree)
- LightGBM: control bằng `num_leaves` thay vì `max_depth`

**Relationship:**
```
max_depth = 6  → balanced tree có 2^6 = 64 leaves
num_leaves = 31 → actual leaves (có thể imbalanced)
```

**Rule of thumb:** `num_leaves < 2^max_depth` để tránh overfit

### Training Process

```python
model = lgb.LGBMRegressor(
    num_leaves=50,
    learning_rate=0.1,
    n_estimators=200
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(10)  # Log mỗi 10 rounds
    ]
)
```

**Mỗi round:**

1. Calculate gradients và hessians
2. Build histogram cho features
3. Find best split cho leaf có highest gain
4. Split leaf → 2 child leaves
5. Repeat cho leaf tiếp theo (cho đến num_leaves)
6. Update predictions
7. Evaluate validation metric

---

## 🎯 Hyperparameter Tuning với Optuna

### Tại sao cần Tuning?

Default hyperparameters **KHÔNG optimal** cho data cụ thể:
- Data khác nhau → Best params khác nhau
- Trade-offs khác nhau (speed vs accuracy)

**Manual tuning:**
- Thử params: `{max_depth: 3, lr: 0.1}` → RMSE = 450
- Thử params: `{max_depth: 5, lr: 0.05}` → RMSE = 420
- Thử params: `{max_depth: 7, lr: 0.1}` → RMSE = 410
- ...

→ Mất nhiều thời gian, không systematic!

### Optuna Bayesian Optimization

**Ý tưởng:**
- Learn từ trials trước để suggest params tốt hơn cho trial sau
- Không phải thử random như Grid Search

**Process:**

```python
def objective(trial):
    # 1. Optuna suggest params từ search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
    }
    
    # 2. Train model với params này
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50)
    
    # 3. Evaluate trên validation
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    # 4. Return metric để minimize
    return rmse

# Run optimization
study = optuna.create_study(
    direction="minimize",                      # Minimize RMSE
    sampler=optuna.samplers.TPESampler(seed=42),  # Bayesian optimization
    pruner=optuna.pruners.MedianPruner()      # Stop bad trials early
)

study.optimize(objective, n_trials=50)
best_params = study.best_params
```

### Optuna Timeline

```
Trial 0:  Params₀ (random)     → RMSE = 450.2
Trial 1:  Params₁ (random)     → RMSE = 435.8
Trial 2:  Params₂ (random)     → RMSE = 442.1
Trial 3:  Params₃ (bayesian)   → RMSE = 428.5 ← Learn từ 0,1,2
Trial 4:  Params₄ (bayesian)   → RMSE = 420.3 ← Better!
...
Trial 15: Params₁₅ (bayesian)  → RMSE = 405.2 ← Best so far
...
Trial 30: Pruned! (không triển vọng)
...
Trial 50: Params₅₀ (bayesian)  → RMSE = 408.1

Best trial: Trial 15 với RMSE = 405.2
```

### TPE Sampler (Tree-structured Parzen Estimator)

**Cách hoạt động:**

1. Chia trials thành 2 groups:
   - **Good trials:** Top 20% với RMSE thấp nhất
   - **Bad trials:** Còn lại

2. Model distributions:
   - `P(params | good)`: Distribution của params trong good trials
   - `P(params | bad)`: Distribution của params trong bad trials

3. Suggest params mới:
   - Chọn params có `P(params | good) / P(params | bad)` cao nhất
   - → Params có probability cao ở good trials, thấp ở bad trials

**Ví dụ:**
```
Good trials: learning_rate thường trong [0.05, 0.15]
Bad trials:  learning_rate thường trong [0.2, 0.3]
→ Suggest learning_rate ≈ 0.1 cho trial tiếp theo
```

### Median Pruner

**Mục đích:** Stop trials không triển vọng sớm để tiết kiệm thời gian

**Logic:**
```python
# Tại mỗi round (e.g., round 50, 100, 150)
current_metric = validation_rmse_at_round_50
median_metric = median(all_completed_trials_at_round_50)

if current_metric > median_metric:
    → Trial này đang worse than median → PRUNE (stop)!
```

**Timeline:**
```
Trial 5 at round 50:  RMSE = 450
Median at round 50:   RMSE = 420
→ 450 > 420 → PRUNE! (không cần train đến round 200)
```

---

## 📊 Evaluation Metrics

### 1. RMSE (Root Mean Squared Error)

```python
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

**Công thức:**
$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_{true,i} - y_{pred,i})^2}$$

**Ý nghĩa:**
- Sai số trung bình theo đơn vị của target
- **Penalty lớn** cho outlier errors (do bình phương)

**Ví dụ:**
```
y_true = [100, 200, 300]
y_pred = [110, 190, 320]
errors = [10, -10, 20]
squared = [100, 100, 400]
MSE = (100+100+400)/3 = 200
RMSE = √200 = 14.14

→ Trung bình sai số ~14 units
```

**Khi nào dùng:**
- ✅ Muốn penalty outliers nhiều hơn
- ✅ Target có outliers cần quan tâm

### 2. MAE (Mean Absolute Error)

```python
mae = mean_absolute_error(y_true, y_pred)
```

**Công thức:**
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_{true,i} - y_{pred,i}|$$

**Ý nghĩa:**
- Sai số tuyệt đối trung bình
- **Linear penalty** (không bình phương)

**So với RMSE:**
```
Same example:
errors = [10, -10, 20]
absolute = [10, 10, 20]
MAE = (10+10+20)/3 = 13.33

MAE < RMSE → RMSE penalty outliers nhiều hơn
```

**Khi nào dùng:**
- ✅ Muốn metric dễ interpret
- ✅ Outliers không quá quan trọng

### 3. MAPE (Mean Absolute Percentage Error)

```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

**Công thức:**
$$MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_{true,i} - y_{pred,i}}{y_{true,i}}\right|$$

**Ý nghĩa:**
- Sai số tính theo **phần trăm** của y_true
- Scale-independent (so sánh được giữa datasets khác scale)

**Ví dụ:**
```
y_true = [100, 1000]
y_pred = [110, 1100]
errors = [10, 100]

MAE = (10+100)/2 = 55 → Không fair!
MAPE = (10/100 + 100/1000)/2 * 100 = (0.1 + 0.1)/2 * 100 = 10%
→ Cả 2 predictions đều sai 10%
```

**⚠️ Cẩn thận:**
- Không dùng khi y_true có giá trị gần 0 (division by zero)
- Asymmetric: over-prediction ít penalty hơn under-prediction

### 4. R² (Coefficient of Determination)

```python
r2 = r2_score(y_true, y_pred)
```

**Công thức:**
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_{true} - y_{pred})^2}{\sum(y_{true} - \bar{y})^2}$$

**Ý nghĩa:**
- **Tỷ lệ variance** của y được model explain
- Range: (-∞, 1]
  - R² = 1: Perfect prediction
  - R² = 0: Model = baseline (predict mean)
  - R² < 0: Model worse than baseline!

**Ví dụ:**
```
y_true = [100, 200, 300, 400]
mean(y_true) = 250

Baseline (predict mean):
SS_tot = (100-250)² + (200-250)² + (300-250)² + (400-250)²
       = 22500 + 2500 + 2500 + 22500 = 50000

Model predictions:
y_pred = [110, 190, 310, 390]
SS_res = (100-110)² + (200-190)² + (300-310)² + (400-390)²
       = 100 + 100 + 100 + 100 = 400

R² = 1 - 400/50000 = 1 - 0.008 = 0.992

→ Model giải thích 99.2% variance!
```

**Khi nào dùng:**
- ✅ So sánh models (R² cao hơn = tốt hơn)
- ✅ Hiểu model fit data tốt đến đâu
- ❌ Không dùng để compare cross datasets (scale-dependent)

### So sánh Metrics

| Metric | Range | Scale | Outlier Sensitivity | Interpretation |
|--------|-------|-------|---------------------|----------------|
| RMSE | [0, ∞) | Same as y | High | Average error in y units |
| MAE | [0, ∞) | Same as y | Low | Average absolute error |
| MAPE | [0, ∞) | Percentage | Medium | % error relative to y |
| R² | (-∞, 1] | Unitless | Medium | % variance explained |

**Trong code:**
```python
def calculate_metrics(y_pred, y_true):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        "r2": r2_score(y_true, y_pred)
    }

# Ví dụ output:
# {
#   "rmse": 245.67,    → Sai số trung bình ~246 sales
#   "mae": 198.32,     → Absolute error ~198 sales  
#   "mape": 12.5,      → Sai số ~12.5%
#   "r2": 0.85         → Explain 85% variance
# }
```

---

## 🔄 Complete Training Flow

### Step-by-step Process

```python
# 1. Load data
sale_processed = pd.read_csv("processed.csv")

# 2. Prepare data (split + feature engineering)
train_df, val_df, test_df = trainer.prepare_data(
    df=sale_processed,
    date_col="date",
    target_col="sales",
    group_cols=["store_id"]
)

# 3. Preprocess features (encode + scale)
X_train, X_val, X_test, y_train, y_val, y_test = \
    trainer.preprocess_features(
        train_df, val_df, test_df,
        target_col="sales",
        exclude_cols=['date']
    )

# 4. Train LightGBM với Optuna tuning
model_lgb = trainer.train_lightgbm_model(
    X_train=X_train,
    X_val=X_val,
    y_train=y_train,
    y_val=y_val,
    use_optuna=True  # Enable hyperparameter tuning
)

# 5. Train XGBoost với Optuna tuning
model_xgb = trainer.train_xgboot_model(
    X_train=X_train,
    X_val=X_val,
    y_train=y_train,
    y_val=y_val,
    use_optuna=True
)

# 6. Evaluate trên test set
result_lgb = model_lgb.predict(X_test)
metrics_lgb = trainer.calculate_metrics(y_test, result_lgb)

result_xgb = model_xgb.predict(X_test)
metrics_xgb = trainer.calculate_metrics(y_test, result_xgb)

# 7. Compare results
print(f"LightGBM - RMSE: {metrics_lgb['rmse']:.2f}, R²: {metrics_lgb['r2']:.4f}")
print(f"XGBoost  - RMSE: {metrics_xgb['rmse']:.2f}, R²: {metrics_xgb['r2']:.4f}")
```

### Timeline Chi tiết

**LightGBM Training:**
```
[Optuna] Starting optimization with 50 trials

Trial 0:
  Params: {num_leaves: 25, learning_rate: 0.15, n_estimators: 120, ...}
  [LightGBM] Training...
  [50]   valid's l2: 0.0450
  [100]  valid's l2: 0.0425
  [120]  valid's l2: 0.0418
  → RMSE = 245.67

Trial 1:
  Params: {num_leaves: 45, learning_rate: 0.08, n_estimators: 180, ...}
  [LightGBM] Training...
  [50]   valid's l2: 0.0435
  [100]  valid's l2: 0.0398
  [142]  valid's l2: 0.0385  ← Best iteration
  [192]  valid's l2: 0.0386  → Early stop
  → RMSE = 238.92

... (48 more trials)

Trial 47:
  Params: {num_leaves: 65, learning_rate: 0.087, n_estimators: 215, ...}
  [LightGBM] Training...
  [142]  valid's l2: 0.0380  ← Best iteration
  → RMSE = 232.18  ← BEST!

Best trial: 47
Best params: {num_leaves: 65, learning_rate: 0.087, ...}
Best RMSE: 232.18

Training final model với best params...
[LightGBM] Final training complete

Test evaluation:
  RMSE: 235.42
  MAE: 189.23
  MAPE: 11.8%
  R²: 0.867
```

**XGBoost Training:** (Tương tự)

---

## 📈 Logs Output Explained

### Optuna Trial Logs

```
[I 2025-12-14 10:30:15,123] Trial 0 finished with value: 245.67 and parameters: {...}
[I 2025-12-14 10:30:18,456] Trial 1 finished with value: 238.92 and parameters: {...}
[I 2025-12-14 10:30:20,234] Trial 2 pruned.
```

- `[I]`: Info log level
- `Trial N`: Trial number (0-indexed)
- `value`: Objective value (RMSE trong case này)
- `parameters`: Hyperparameters được thử
- `pruned`: Trial bị stop sớm vì không triển vọng

### LightGBM Training Logs

```
[LightGBM] [Info] Training until validation scores don't improve for 50 rounds
[50]    valid_0's l2: 0.0423
[100]   valid_0's l2: 0.0398
[150]   valid_0's l2: 0.0385
[200]   valid_0's l2: 0.0386
Early stopping, best iteration is [142]
```

- `[N]`: Boosting round number
- `valid_0's l2`: MSE loss trên validation set
  - l2 = MSE (mean squared error)
  - Giảm dần = model đang học
- `Early stopping`: Triggered vì 50 rounds không cải thiện
- `best iteration [142]`: Round 142 có loss thấp nhất

### XGBoost Training Logs

```
[0]     validation_0-rmse:450.23
[50]    validation_0-rmse:398.45
[100]   validation_0-rmse:385.12
[150]   validation_0-rmse:382.67
[200]   validation_0-rmse:383.15
Stopping. Best iteration:
[157]   validation_0-rmse:382.45
```

- `[N]`: Boosting round
- `validation_0-rmse`: RMSE trên validation set
- `Best iteration [157]`: Round có RMSE thấp nhất

---

## ⚙️ Configuration

```yaml
# config.yaml
training:
  train_size: 0.7        # 70% cho training
  val_size: 0.15         # 15% cho validation
  early_stop: 50         # Stop nếu 50 rounds không improve
  optuna_trials: 50      # Số trials cho hyperparameter tuning

models:
  xgboost:
    params:              # Default params (nếu không dùng Optuna)
      n_estimators: 200
      max_depth: 6
      learning_rate: 0.1
      subsample: 0.8
      colsample_bytree: 0.8
      
  lightgbm:
    params:
      num_leaves: 31
      learning_rate: 0.1
      n_estimators: 200
      min_child_samples: 20
```

---

## 🎓 Best Practices

### 1. Data Splitting
- ✅ Split theo **thời gian** cho time series
- ✅ Train trên data cũ, test trên data mới
- ❌ Không shuffle time series data

### 2. Validation Set
- ✅ Luôn dùng validation set riêng cho early stopping
- ✅ Validation set phải representative cho test set
- ❌ Không tune trên test set (data leakage!)

### 3. Hyperparameter Tuning
- ✅ Dùng Optuna để tiết kiệm thời gian
- ✅ Set reasonable search ranges
- ✅ Start với ít trials (20) để test, tăng dần (50-100)
- ❌ Không set range quá rộng (lãng phí trials)

### 4. Overfitting Prevention
- ✅ Monitor validation metrics
- ✅ Use early stopping
- ✅ Use regularization (L1/L2, gamma)
- ✅ Reduce max_depth/num_leaves nếu thấy overfit
- ⚠️ Train loss << Val loss = Overfitting signal!

### 5. Model Comparison
- ✅ Compare trên **same test set**
- ✅ Look at multiple metrics (RMSE, MAE, R²)
- ✅ Consider training time vs accuracy trade-off
- ✅ Ensemble thường tốt hơn single model

---

## 🐛 Troubleshooting

### Training quá chậm

**Nguyên nhân:**
- Quá nhiều Optuna trials
- `n_estimators` quá lớn
- Dataset quá lớn

**Solutions:**
```python
# Giảm trials
optuna_trials: 50 → 20

# Giảm n_estimators range
"n_estimators": trial.suggest_int("n_estimators", 50, 150)  # Thay vì 300

# Subsample data
X_train_sample = X_train.sample(frac=0.5)  # Dùng 50% data
```

### Overfitting (Train loss << Val loss)

**Nguyên nhân:**
- Model quá complex
- Regularization yếu
- Train quá nhiều rounds

**Solutions:**
```python
# Tăng regularization
"reg_alpha": trial.suggest_float("reg_alpha", 0.5, 2.0)  # Tăng từ [0, 1.0]
"reg_lambda": trial.suggest_float("reg_lambda", 1.0, 3.0)

# Giảm complexity
"max_depth": trial.suggest_int("max_depth", 3, 6)  # Thay vì 3-10
"num_leaves": trial.suggest_int("num_leaves", 20, 50)  # Thay vì 20-100

# Aggressive early stopping
early_stopping_rounds = 30  # Thay vì 50
```

### Underfitting (Cả train và val loss đều cao)

**Nguyên nhân:**
- Model quá simple
- Learning rate quá thấp
- Không đủ features

**Solutions:**
```python
# Tăng complexity
"max_depth": trial.suggest_int("max_depth", 6, 12)
"num_leaves": trial.suggest_int("num_leaves", 50, 150)

# Tăng learning rate
"learning_rate": trial.suggest_float("learning_rate", 0.05, 0.3)

# Add more features
# → Check feature engineering pipeline
```

### Val RMSE tốt nhưng Test RMSE kém

**Nguyên nhân:**
- Validation set không representative
- Overfit trên validation set (tune quá nhiều)

**Solutions:**
- Tăng validation set size
- K-fold cross validation
- Re-split data với seed khác

---

## 📚 References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Gradient Boosting Explained](https://explained.ai/gradient-boosting/)
