# MÃ GIẢ:  HỆ THỐNG TRAINING MÔ HÌNH DỰ BÁO BÁN HÀNG

## 1. TỔNG QUAN KIẾN TRÚC

```
┌─────────────────────────────────────────────────────────────┐
│                    ModelTrainer                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  CONFIG & DEPENDENCIES                                │   │
│  │  - MLflow Manager (tracking experiments)             │   │
│  │  - Feature Engineer (tạo features)                   │   │
│  │  - Data Validator (kiểm tra data)                    │   │
│  │  - Models, Scalers, Encoders                         │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  DATA PREPARATION PIPELINE                            │   │
│  │  1. Validate essential columns                       │   │
│  │  2. Feature engineering                              │   │
│  │  3. Chronological split (train/val/test)            │   │
│  │  4. Clean NaN values                                 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  PREPROCESSING PIPELINE                               │   │
│  │  1. Separate features & target                       │   │
│  │  2. Encode categorical variables (LabelEncoder)      │   │
│  │  3. Scale numerical features (StandardScaler)        │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  MODEL TRAINING                                       │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐     │   │
│  │  │  XGBoost   │  │  LightGBM  │  │  Prophet   │     │   │
│  │  │  + Optuna  │  │  + Optuna  │  │ (optional) │     │   │
│  │  └────────────┘  └────────────┘  └────────────┘     │   │
│  │           │              │              │            │   │
│  │           └──────────────┴──────────────┘            │   │
│  │                      │                               │   │
│  │              ┌───────▼────────┐                      │   │
│  │              │  ENSEMBLE      │                      │   │
│  │              │  (weighted avg)│                      │   │
│  │              └────────────────┘                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  EVALUATION & LOGGING                                 │   │
│  │  - Metrics:  RMSE, MAE, MAPE, R2                      │   │
│  │  - Visualizations                                     │   │
│  │  - MLflow tracking                                    │   │
│  │  - S3 artifact sync                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. CLASS ModelTrainer - KHỞI TẠO

```pseudocode
CLASS ModelTrainer:
    
    FUNCTION __init__(config_path):
        """
        Mục đích: Khởi tạo trainer với tất cả dependencies
        
        Input:  config_path - đường dẫn file YAML config
        Output: ModelTrainer instance đã setup
        """
        
        # Bước 1: Load configuration từ YAML
        config = ĐỌC_FILE_YAML(config_path)
        
        # Bước 2: Extract các config sections
        model_config = config['models']           # Hyperparameters cho từng model
        training_config = config['training']      # Train/val/test split ratios
        
        # Bước 3: Khởi tạo các manager/helper objects
        mlflow_manager = TẠO_MLflowManager(config_path)
        feature_engineer = TẠO_FeatureEngineer(config_path)
        data_validator = TẠO_DataValidator(config_path)
        
        # Bước 4: Khởi tạo dictionaries lưu trữ
        models = {}        # Lưu trained models
        scalers = {}       # Lưu StandardScaler objects
        encoders = {}      # Lưu LabelEncoder objects
        
        LƯU_TẤT_CẢ_VÀO_SELF()
    END FUNCTION
```

---

## 3. DATA PREPARATION PIPELINE

```pseudocode
FUNCTION prepare_data(df, target_col, date_col, group_cols, categorical_cols):
    """
    Mục đích: Chuẩn bị data cho training (time-series aware)
    
    Ý tưởng chính:
    - Time series data cần split chronologically (không random)
    - Validation & test set nên là data gần nhất (gần với thực tế predict)
    - Feature engineering tạo lag, rolling, seasonal features
    
    Input: 
        df - DataFrame gốc
        target_col - tên cột target (vd: 'sales')
        date_col - tên cột date
        group_cols - các cột để group (vd: ['store_id'])
        categorical_cols - các cột categorical cần encode
    
    Output: 
        train_df, val_df, test_df - 3 DataFrames đã split
    """
    
    LOG("Bắt đầu chuẩn bị data cho training")
    
    # ============================================================
    # BƯỚC 1: VALIDATION - Kiểm tra columns cần thiết
    # ============================================================
    
    required_cols = ['date', target_col]
    NẾU group_cols TỒN_TẠI:
        THÊM group_cols VÀO required_cols
    
    missing_cols = TÌM_COLUMNS_THIẾU(df. columns, required_cols)
    
    NẾU missing_cols KHÔNG RỖNG:
        THROW_ERROR("Thiếu columns: {missing_cols}")
    
    # ============================================================
    # BƯỚC 2: FEATURE ENGINEERING
    # ============================================================
    
    """
    Feature engineering tạo các features mới:
    - Lag features:  sales_lag_7, sales_lag_30 (giá trị bán hàng trước đó)
    - Rolling features:  sales_rolling_mean_7, sales_rolling_std_30
    - Time features: day_of_week, month, quarter, is_weekend
    - Seasonal features: sales_lag_365 (cùng kỳ năm trước)
    """
    
    df_features = feature_engineer.create_all_features(
        df, 
        target_col, 
        date_col, 
        group_cols, 
        categorical_cols
    )
    
    # ============================================================
    # BƯỚC 3: CHRONOLOGICAL SPLIT (Quan trọng với time series!)
    # ============================================================
    
    """
    Tại sao split chronologically? 
    - Time series có tính tuần tự:  quá khứ → hiện tại → tương lai
    - Không thể dùng random split vì sẽ bị "data leakage"
    - Model phải học từ quá khứ và predict tương lai
    
    Tại sao validation/test là data gần nhất?
    - Model sẽ predict tương lai gần
    - Data gần nhất phản ánh xu hướng hiện tại tốt nhất
    - Tránh overfitting vào patterns cũ
    """
    
    # Sắp xếp theo thời gian
    df_sorted = SẮP_XẾP(df_features, THEO=date_col, TĂNG_DẦN=True)
    
    # Tính toán kích thước mỗi split
    total_size = ĐẾM_ROWS(df_sorted)
    test_ratio = training_config['test_size']           # vd: 0.15
    val_ratio = training_config['validation_size']      # vd: 0.15
    
    train_size = total_size * (1 - test_ratio - val_ratio)  # 70%
    val_size = total_size * val_ratio                        # 15%
    test_size = total_size * test_ratio                      # 15%
    
    # Split data
    train_df = df_sorted[0 : train_size]
    val_df = df_sorted[train_size : train_size + val_size]
    test_df = df_sorted[train_size + val_size : ]
    
    """
    Visualization của split:
    
    |===============================================|
    |        PAST        | RECENT | LATEST         |
    |--------------------+--------+----------------|
    |   Training (70%)   | Val    | Test          |
    |                    | (15%)  | (15%)         |
    |===============================================|
                              ↑           ↑
                         Validate     Predict
                         trên đây    tương lai
    """
    
    # ============================================================
    # BƯỚC 4: CLEAN NaN VALUES
    # ============================================================
    
    """
    Tại sao có NaN? 
    - Lag features tạo NaN ở đầu (không có giá trị trước đó)
    - Rolling features cần window nên đầu series có NaN
    
    Chiến lược:
    - Chỉ xóa rows có NaN ở target column (không thể train)
    - NaN ở features có thể handle bằng model hoặc imputation
    """
    
    train_df = XÓA_ROWS_CÓ_NaN(train_df, TRONG_COLUMN=target_col)
    val_df = XÓA_ROWS_CÓ_NaN(val_df, TRONG_COLUMN=target_col)
    test_df = XÓA_ROWS_CÓ_NaN(test_df, TRONG_COLUMN=target_col)
    
    # ============================================================
    # BƯỚC 5: LOGGING & RETURN
    # ============================================================
    
    LOG(f"Data split - Train: {ĐẾM(train_df)}, Val: {ĐẾM(val_df)}, Test: {ĐẾM(test_df)}")
    
    RETURN train_df, val_df, test_df
END FUNCTION
```

---

## 4. PREPROCESSING PIPELINE

```pseudocode
FUNCTION preprocess_features(train_df, val_df, test_df, target_col, exclude_cols):
    """
    Mục đích: Transform features thành format cho ML models
    
    Ý tưởng: 
    - Separate X (features) và y (target)
    - Encode categorical → numbers
    - Scale numerical → chuẩn hóa về cùng scale
    
    Input:  train/val/test DataFrames
    Output: X_train, X_val, X_test, y_train, y_val, y_test (arrays)
    """
    
    # ============================================================
    # BƯỚC 1: SEPARATE FEATURES & TARGET
    # ============================================================
    
    """
    Loại trừ: 
    - date column (không phải feature)
    - target column
    - các exclude_cols khác
    """
    
    feature_cols = []
    CHO MỖI column TRONG train_df. columns:
        NẾU column KHÔNG_TRONG [exclude_cols, target_col]:
            THÊM column VÀO feature_cols
    
    # Extract features
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Extract target
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]
    
    # ============================================================
    # BƯỚC 2: ENCODE CATEGORICAL VARIABLES
    # ============================================================
    
    """
    Tại sao cần encode? 
    - ML models chỉ hiểu numbers, không hiểu text
    - LabelEncoder:  'Monday' → 0, 'Tuesday' → 1, ... 
    
    Quan trọng:  FIT trên train, TRANSFORM trên val/test
    - Tránh data leakage
    - Đảm bảo consistency
    """
    
    categorical_cols = TÌM_CATEGORICAL_COLUMNS(X_train)
    
    CHO MỖI col TRONG categorical_cols:
        
        NẾU col CHƯA CÓ encoder:
            # Tạo encoder mới và fit trên training data
            encoders[col] = TẠO_LabelEncoder()
            X_train[col] = encoders[col].FIT_TRANSFORM(X_train[col])
        NGƯỢC_LẠI:
            # Dùng encoder đã có (cho consistency)
            X_train[col] = encoders[col]. TRANSFORM(X_train[col])
        
        # Transform validation và test với CÙNG encoder
        X_val[col] = encoders[col].TRANSFORM(X_val[col])
        X_test[col] = encoders[col]. TRANSFORM(X_test[col])
    
    # ============================================================
    # BƯỚC 3: SCALE NUMERICAL FEATURES
    # ============================================================
    
    """
    Tại sao cần scaling?
    - Features có scales khác nhau:  sales (1000s) vs day_of_week (1-7)
    - Gradient descent học tốt hơn khi features cùng scale
    - Một số models (SVM, Neural Nets) yêu cầu scaling
    
    StandardScaler: (x - mean) / std
    - Mean = 0, Std = 1
    - Giữ shape của distribution
    """
    
    scaler = TẠO_StandardScaler()
    
    # Fit trên training data
    X_train_scaled = scaler.FIT_TRANSFORM(X_train)
    
    # Transform validation và test với CÙNG scaler
    X_val_scaled = scaler.TRANSFORM(X_val)
    X_test_scaled = scaler.TRANSFORM(X_test)
    
    # ============================================================
    # BƯỚC 4: CONVERT BACK TO DATAFRAME
    # ============================================================
    
    """
    Tại sao convert lại DataFrame?
    - Giữ feature names (quan trọng cho feature importance)
    - Dễ debug và visualize
    - Một số models cần column names
    """
    
    X_train_scaled = CHUYỂN_VỀ_DATAFRAME(X_train_scaled, 
                                          columns=feature_cols,
                                          index=X_train.index)
    
    X_val_scaled = CHUYỂN_VỀ_DATAFRAME(X_val_scaled,
                                        columns=feature_cols,
                                        index=X_val.index)
    
    X_test_scaled = CHUYỂN_VỀ_DATAFRAME(X_test_scaled,
                                         columns=feature_cols,
                                         index=X_test.index)
    
    # ============================================================
    # BƯỚC 5: SAVE & RETURN
    # ============================================================
    
    LƯU(scaler, VÀO=scalers['standard'])
    LƯU(feature_cols, VÀO=self.feature_cols)
    
    RETURN X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
END FUNCTION
```

---

## 5. METRICS CALCULATION

```pseudocode
FUNCTION calculate_metrics(y_true, y_pred):
    """
    Mục đích: Tính các metrics đánh giá model
    
    Metrics:
    1.  RMSE (Root Mean Squared Error)
       - Đơn vị giống target
       - Phạt nặng outliers
       - RMSE = sqrt(mean((y_true - y_pred)²))
    
    2. MAE (Mean Absolute Error)
       - Đơn vị giống target
       - Không phạt nặng outliers
       - MAE = mean(|y_true - y_pred|)
    
    3. MAPE (Mean Absolute Percentage Error)
       - Đơn vị %
       - Dễ hiểu cho business
       - MAPE = mean(|y_true - y_pred| / y_true) * 100
    
    4. R² (R-squared / Coefficient of Determination)
       - Từ 0 đến 1 (1 là perfect)
       - Tỷ lệ variance được giải thích
       - R² = 1 - (SS_res / SS_tot)
    """
    
    metrics = {}
    
    # RMSE
    mse = MEAN((y_true - y_pred)²)
    metrics['rmse'] = SQRT(mse)
    
    # MAE
    metrics['mae'] = MEAN(ABS(y_true - y_pred))
    
    # MAPE
    percentage_errors = ABS((y_true - y_pred) / y_true)
    metrics['mape'] = MEAN(percentage_errors) * 100
    
    # R²
    ss_res = SUM((y_true - y_pred)²)      # Residual sum of squares
    ss_tot = SUM((y_true - MEAN(y_true))²) # Total sum of squares
    metrics['r2'] = 1 - (ss_res / ss_tot)
    
    RETURN metrics
END FUNCTION
```

---

## 6. TRAIN XGBOOST WITH OPTUNA

```pseudocode
FUNCTION train_xgboost(X_train, y_train, X_val, y_val, use_optuna):
    """
    Mục đích: Train XGBoost model với hyperparameter tuning
    
    XGBoost là gì?
    - Gradient Boosting Decision Trees
    - Build trees sequentially, mỗi tree fix lỗi của trees trước
    - Rất mạnh cho tabular data
    
    Optuna là gì?
    - Automated hyperparameter optimization
    - Thử nhiều combinations, tìm best params
    - Dùng Bayesian optimization (thông minh hơn grid search)
    """
    
    LOG("Bắt đầu training XGBoost")
    
    # ============================================================
    # OPTUNA HYPERPARAMETER TUNING (nếu enabled)
    # ============================================================
    
    NẾU use_optuna == True:
        
        # Định nghĩa objective function
        FUNCTION objective(trial):
            """
            Mục đích:  Function Optuna sẽ tối ưu
            
            Input:  trial - một lần thử với một set params
            Output: metric cần minimize (RMSE)
            """
            
            # Suggest hyperparameters trong range cho phép
            params = {
                'n_estimators': trial. SUGGEST_INT('n_estimators', 50, 300),
                # Số trees - nhiều = phức tạp hơn nhưng risk overfit
                
                'max_depth': trial. SUGGEST_INT('max_depth', 3, 10),
                # Độ sâu mỗi tree - sâu = capture details hơn
                
                'learning_rate': trial.SUGGEST_FLOAT('learning_rate', 0.01, 0.3),
                # Tốc độ học - thấp = chậm nhưng stable
                
                'subsample': trial.SUGGEST_FLOAT('subsample', 0.6, 1.0),
                # % samples dùng cho mỗi tree - < 1 giúp prevent overfit
                
                'colsample_bytree': trial. SUGGEST_FLOAT('colsample_bytree', 0.6, 1.0),
                # % features dùng cho mỗi tree
                
                'gamma': trial.SUGGEST_FLOAT('gamma', 0, 0.5),
                # Minimum loss reduction - cao = conservative
                
                'reg_alpha': trial.SUGGEST_FLOAT('reg_alpha', 0, 1. 0),
                # L1 regularization
                
                'reg_lambda':  trial.SUGGEST_FLOAT('reg_lambda', 0, 1.0),
                # L2 regularization
                
                'random_state': 42  # For reproducibility
            }
            
            # Train model với params này
            model = TẠO_XGBRegressor(params)
            model.FIT(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50)
            
            # Predict và tính RMSE
            y_pred = model.PREDICT(X_val)
            rmse = SQRT(MEAN((y_val - y_pred)²))
            
            RETURN rmse  # Optuna sẽ minimize cái này
        END FUNCTION
        
        # Tạo Optuna study
        study = optuna.CREATE_STUDY(
            direction='minimize',  # Minimize RMSE
            sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
            pruner=MedianPruner()  # Stop bad trials sớm
        )
        
        # Chạy optimization
        n_trials = config['training']['optuna_trials']  # vd: 50 trials
        study. OPTIMIZE(objective, n_trials=n_trials)
        
        # Lấy best parameters
        best_params = study. best_params
        best_params['random_state'] = 42
        
    NGƯỢC_LẠI: 
        # Dùng default params từ config
        best_params = model_config['xgboost']['params']
    
    # ============================================================
    # TRAIN FINAL MODEL VỚI BEST PARAMS
    # ============================================================
    
    best_params['early_stopping_rounds'] = 50
    
    model = TẠO_XGBRegressor(**best_params)
    model.FIT(X_train, y_train,
             eval_set=[(X_val, y_val)],
             verbose=True)
    
    # Lưu model
    models['xgboost'] = model
    
    RETURN model
END FUNCTION
```

---

## 7. TRAIN LIGHTGBM WITH OPTUNA

```pseudocode
FUNCTION train_lightgbm(X_train, y_train, X_val, y_val, use_optuna):
    """
    Mục đích: Train LightGBM model
    
    LightGBM vs XGBoost:
    - Cũng là Gradient Boosting nhưng faster
    - Leaf-wise growth (vs level-wise của XGBoost)
    - Better với large datasets
    - Thường cho kết quả tương đương hoặc tốt hơn XGBoost
    
    Logic tương tự XGBoost
    """
    
    LOG("Bắt đầu training LightGBM")
    
    NẾU use_optuna == True:
        
        FUNCTION objective(trial):
            params = {
                'num_leaves': trial.SUGGEST_INT('num_leaves', 20, 100),
                # Số leaves - LightGBM specific param
                
                'learning_rate': trial.SUGGEST_FLOAT('learning_rate', 0.01, 0.3),
                'n_estimators': trial.SUGGEST_INT('n_estimators', 50, 300),
                'min_child_samples': trial.SUGGEST_INT('min_child_samples', 10, 50),
                # Minimum samples trong leaf
                
                'subsample': trial.SUGGEST_FLOAT('subsample', 0.6, 1.0),
                'colsample_bytree': trial.SUGGEST_FLOAT('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.SUGGEST_FLOAT('reg_alpha', 0, 1.0),
                'reg_lambda': trial. SUGGEST_FLOAT('reg_lambda', 0, 1.0),
                
                'random_state': 42,
                'verbosity': -1,
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type':  'gbdt'
            }
            
            model = TẠO_LGBMRegressor(**params)
            model.FIT(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     callbacks=[early_stopping(50)])
            
            y_pred = model.PREDICT(X_val)
            rmse = SQRT(MEAN((y_val - y_pred)²))
            
            RETURN rmse
        END FUNCTION
        
        study = optuna.CREATE_STUDY(direction='minimize')
        study. OPTIMIZE(objective, n_trials=n_trials)
        
        best_params = study.best_params
    NGƯỢC_LẠI: 
        best_params = model_config['lightgbm']['params']
    
    # Train final model
    model = TẠO_LGBMRegressor(**best_params)
    model.FIT(X_train, y_train,
             eval_set=[(X_val, y_val)],
             callbacks=[early_stopping(50)])
    
    models['lightgbm'] = model
    
    RETURN model
END FUNCTION
```

---

## 8. TRAIN PROPHET

```pseudocode
FUNCTION train_prophet(train_df, val_df, date_col, target_col):
    """
    Mục đích: Train Facebook Prophet model
    
    Prophet là gì?
    - Time series forecasting model từ Facebook
    - Tự động detect seasonality (yearly, weekly, daily)
    - Handle holidays và special events
    - Robust với missing data
    - Khác XGBoost/LightGBM:  không cần feature engineering nhiều
    
    Prophet format:
    - Cần 2 columns: 'ds' (date) và 'y' (target)
    - Có thể add thêm regressors (như features)
    """
    
    LOG("Bắt đầu training Prophet")
    
    # ============================================================
    # BƯỚC 1: PREPARE DATA CHO PROPHET
    # ============================================================
    
    # Prophet yêu cầu columns 'ds' và 'y'
    prophet_train = train_df[[date_col, target_col]]
    prophet_train = ĐỔI_TÊN_COLUMNS(prophet_train, {
        date_col: 'ds',
        target_col: 'y'
    })
    
    # Remove NaN và sort theo date
    prophet_train = XÓA_NaN(prophet_train)
    prophet_train = SẮP_XẾP(prophet_train, 'ds')
    
    # ============================================================
    # BƯỚC 2: CONFIGURE PROPHET
    # ============================================================
    
    """
    Prophet parameters:
    - yearly_seasonality:  Có pattern theo năm?  (vd: mua hè/đông)
    - weekly_seasonality: Có pattern theo tuần?  (vd: cuối tuần)
    - daily_seasonality: Có pattern theo ngày? (vd: giờ cao điểm)
    - changepoint_prior_scale: Flexibility cho trend changes
    - seasonality_prior_scale: Strength của seasonality
    """
    
    prophet_params = model_config['prophet']['params']
    
    # Override một số params cho stability
    prophet_params['stan_backend'] = 'CMDSTANPY'
    prophet_params['mcmc_samples'] = 0  # Disable MCMC (faster)
    prophet_params['uncertainty_samples'] = 100
    
    TRY:
        model = TẠO_Prophet(**prophet_params)
        
        # ============================================================
        # BƯỚC 3: ADD REGRESSORS (Optional features)
        # ============================================================
        
        """
        Regressors = external features ảnh hưởng đến target
        Vd: promotions, holidays, weather, competitor actions
        """
        
        numeric_cols = TÌM_NUMERIC_COLUMNS(train_df)
        
        # Loại trừ các columns không phù hợp
        exclude = [target_col, 'year', 'month', 'day', 'week', 'quarter']
        regressor_cols = LỌC(numeric_cols, KHÔNG_TRONG=exclude)
        
        # Chỉ lấy top 5 regressors quan trọng nhất (by variance)
        NẾU ĐẾM(regressor_cols) > 5:
            variances = {}
            CHO MỖI col TRONG regressor_cols:
                variances[col] = TÍNH_VARIANCE(train_df[col])
            
            regressor_cols = SẮP_XẾP(regressor_cols, 
                                      THEO=variances, 
                                      GIẢM_DẦN=True)[:5]
        
        # Add regressors vào model
        CHO MỖI col TRONG regressor_cols: 
            NẾU VARIANCE(train_df[col]) > 0:
                model.ADD_REGRESSOR(col)
                prophet_train[col] = train_df[col]
        
        # ============================================================
        # BƯỚC 4: FIT MODEL
        # ============================================================
        
        model.FIT(prophet_train)
        
        models['prophet'] = model
        RETURN model
        
    EXCEPT Exception e:
        LOG_ERROR(f"Prophet failed: {e}")
        
        # Retry với minimal config
        LOG("Retry Prophet với config đơn giản...")
        
        model = TẠO_Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            uncertainty_samples=50,
            mcmc_samples=0
        )
        
        # Train KHÔNG CÓ regressors
        model.FIT(prophet_train[['ds', 'y']])
        
        models['prophet'] = model
        RETURN model
    END TRY
END FUNCTION
```

---

## 9. TRAIN ALL MODELS (MAIN ORCHESTRATOR)

```pseudocode
FUNCTION train_all_models(train_df, val_df, test_df, target_col, use_optuna):
    """
    Mục đích:  Orchestrate toàn bộ training pipeline
    
    Flow:
    1. Start MLflow tracking
    2. Preprocess data
    3. Train individual models
    4. Create ensemble
    5. Evaluate all models
    6. Log results & artifacts
    7.  Sync to S3
    """
    
    results = {}  # Lưu kết quả từng model
    
    # ============================================================
    # BƯỚC 1: START MLFLOW RUN
    # ============================================================
    
    """
    MLflow là gì?
    - Platform tracking ML experiments
    - Log:  parameters, metrics, models, artifacts
    - So sánh runs, reproduce results
    - Deploy models
    """
    
    run_name = f"sales_forecast_training_{DATETIME_NOW()}"
    run_id = mlflow_manager.START_RUN(
        run_name=run_name,
        tags={"model_type": "ensemble", "use_optuna": use_optuna}
    )
    
    TRY:
        # ============================================================
        # BƯỚC 2: PREPROCESS DATA
        # ============================================================
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocess_features(
            train_df, val_df, test_df, target_col
        )
        
        # Log data stats vào MLflow
        mlflow_manager.LOG_PARAMS({
            "train_size": ĐẾM(train_df),
            "val_size": ĐẾM(val_df),
            "test_size": ĐẾM(test_df),
            "n_features": X_train.shape[1]
        })
        
        # ============================================================
        # BƯỚC 3: TRAIN XGBOOST
        # ============================================================
        
        xgb_model = train_xgboost(X_train, y_train, X_val, y_val, use_optuna)
        
        # Predict trên test set
        xgb_pred = xgb_model. PREDICT(X_test)
        
        # Calculate metrics
        xgb_metrics = calculate_metrics(y_test, xgb_pred)
        
        # Log vào MLflow
        mlflow_manager.LOG_METRICS({
            f"xgboost_{k}": v for k, v in xgb_metrics
        })
        mlflow_manager.LOG_MODEL(xgb_model, "xgboost")
        
        # Log feature importance
        feature_importance = TẠO_DATAFRAME({
            'feature': feature_cols,
            'importance': xgb_model.feature_importances_
        })
        feature_importance = SẮP_XẾP(feature_importance, 
                                      'importance', 
                                      GIẢM_DẦN=True)[:20]
        
        LOG(f"Top XGBoost features:\n{feature_importance}")
        
        # Lưu vào results
        results['xgboost'] = {
            'model':  xgb_model,
            'metrics': xgb_metrics,
            'predictions': xgb_pred
        }
        
        # ============================================================
        # BƯỚC 4: TRAIN LIGHTGBM
        # ============================================================
        
        lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, use_optuna)
        lgb_pred = lgb_model.PREDICT(X_test)
        lgb_metrics = calculate_metrics(y_test, lgb_pred)
        
        mlflow_manager.LOG_METRICS({
            f"lightgbm_{k}": v for k, v in lgb_metrics
        })
        mlflow_manager.LOG_MODEL(lgb_model, "lightgbm")
        
        results['lightgbm'] = {
            'model': lgb_model,
            'metrics': lgb_metrics,
            'predictions': lgb_pred
        }
        
        # ============================================================
        # BƯỚC 5: TRAIN PROPHET (Optional)
        # ============================================================
        
        prophet_enabled = model_config['prophet']['enabled']
        
        NẾU prophet_enabled == True:
            TRY:
                prophet_model = train_prophet(train_df, val_df)
                
                # Prepare future dataframe cho predictions
                future = test_df[['date']]. RENAME({'date': 'ds'})
                
                # Add regressors nếu model có
                NẾU prophet_model CÓ_REGRESSORS: 
                    CHO MỖI col TRONG prophet_model.regressors:
                        NẾU col TRONG test_df.columns:
                            future[col] = test_df[col]
                
                # Predict
                forecast = prophet_model.PREDICT(future)
                prophet_pred = forecast['yhat']  # yhat = predicted value
                
                prophet_metrics = calculate_metrics(y_test, prophet_pred)
                
                mlflow_manager. LOG_METRICS({
                    f"prophet_{k}": v for k, v in prophet_metrics
                })
                
                results['prophet'] = {
                    'model': prophet_model,
                    'metrics': prophet_metrics,
                    'predictions': prophet_pred
                }
                
            EXCEPT Exception e:
                LOG_WARNING(f"Prophet failed: {e}. Sẽ dùng ensemble 2 models.")
                prophet_enabled = False
            END TRY
        END IF
        
        # ============================================================
        # BƯỚC 6: CREATE ENSEMBLE MODEL
        # ============================================================
        
        """
        Ensemble là gì?
        - Kết hợp predictions từ nhiều models
        - Thường tốt hơn individual models
        - Reduce variance, more robust
        
        Strategies:
        1. Simple average:  (pred1 + pred2 + pred3) / 3
        2. Weighted average: w1*pred1 + w2*pred2 + w3*pred3
        3. Stacking: Train meta-model trên predictions
        
        Code này dùng weighted average based on validation R²
        """
        
        NẾU prophet_enabled == True:
            # Ensemble 3 models:  XGBoost + LightGBM + Prophet
            ensemble_pred = (xgb_pred + lgb_pred + prophet_pred) / 3
            
            ensemble_models = {
                'xgboost': xgb_model,
                'lightgbm':  lgb_model,
                'prophet': prophet_model
            }
            
            ensemble_weights = {
                'xgboost': 1/3,
                'lightgbm': 1/3,
                'prophet': 1/3
            }
            
        NGƯỢC_LẠI: 
            # Weighted ensemble 2 models based on validation performance
            
            # Calculate validation predictions
            xgb_val_pred = xgb_model. PREDICT(X_val)
            lgb_val_pred = lgb_model.PREDICT(X_val)
            
            # Calculate R² scores
            xgb_val_r2 = CALCULATE_R2(y_val, xgb_val_pred)
            lgb_val_r2 = CALCULATE_R2(y_val, lgb_val_pred)
            
            """
            Ý tưởng weights:
            - Model có R² cao hơn → weight lớn hơn
            - Minimum weight = 0.2 (không ignore model hoàn toàn)
            - Normalize về tổng = 1
            
            Example:
            - xgb_r2 = 0.85, lgb_r2 = 0.80
            - xgb_weight = 0.85/(0.85+0.80) = 0.515
            - lgb_weight = 0.80/(0.85+0.80) = 0.485
            """
            
            min_weight = 0.2
            
            xgb_weight = MAX(min_weight, xgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
            lgb_weight = MAX(min_weight, lgb_val_r2 / (xgb_val_r2 + lgb_val_r2))
            
            # Normalize weights
            total_weight = xgb_weight + lgb_weight
            xgb_weight = xgb_weight / total_weight
            lgb_weight = lgb_weight / total_weight
            
            LOG(f"Ensemble weights - XGBoost: {xgb_weight:.3f}, LightGBM:  {lgb_weight:.3f}")
            
            # Calculate ensemble predictions
            ensemble_pred = xgb_weight * xgb_pred + lgb_weight * lgb_pred
            
            ensemble_models = {
                'xgboost': xgb_model,
                'lightgbm': lgb_model
            }
            
            ensemble_weights = {
                'xgboost': xgb_weight,
                'lightgbm': lgb_weight
            }
        END IF
        
        # Create ensemble model object
        ensemble_model = TẠO_EnsembleModel(ensemble_models, ensemble_weights)
        
        # Calculate ensemble metrics
        ensemble_metrics = calculate_metrics(y_test, ensemble_pred)
        
        # Log ensemble
        mlflow_manager.LOG_METRICS({
            f"ensemble_{k}": v for k, v in ensemble_metrics
        })
        mlflow_manager.LOG_MODEL(ensemble_model, "ensemble")
        
        results['ensemble'] = {
            'model': ensemble_model,
            'metrics': ensemble_metrics,
            'predictions': ensemble_pred
        }
        
        # ============================================================
        # BƯỚC 7: RUN DIAGNOSTICS
        # ============================================================
        
        """
        Diagnostics kiểm tra: 
        - Overfitting?  (train metrics >> test metrics)
        - Underfitting? (cả train và test metrics đều kém)
        - Residuals patterns?  (errors có pattern hay random?)
        - Feature issues? (multicollinearity, leakage?)
        
        Output:  Recommendations để improve model
        """
        
        LOG("Running model diagnostics...")
        
        test_predictions = {
            'xgboost':  xgb_pred,
            'lightgbm': lgb_pred,
            'ensemble': ensemble_pred
        }
        
        diagnosis = diagnose_model_performance(
            train_df, val_df, test_df, 
            test_predictions, target_col
        )
        
        LOG("Diagnostic recommendations:")
        CHO MỖI recommendation TRONG diagnosis['recommendations']:
            LOG_WARNING(f"- {recommendation}")
        
        # ============================================================
        # BƯỚC 8: GENERATE VISUALIZATIONS
        # ============================================================
        
        """
        Visualizations:
        - Metrics comparison bar chart
        - Predictions vs Actual timeline
        - Residuals plot
        - Error distribution histogram
        - Feature importance charts
        - Summary statistics table
        """
        
        LOG("Generating visualizations...")
        
        TRY:
            generate_and_log_visualizations(results, test_df, target_col)
        EXCEPT Exception e:
            LOG_ERROR(f"Visualization failed: {e}")
        END TRY
        
        # ============================================================
        # BƯỚC 9: SAVE ARTIFACTS
        # ============================================================
        
        """
        Artifacts cần save:
        - Models (xgboost, lightgbm, prophet, ensemble)
        - Scalers (StandardScaler)
        - Encoders (LabelEncoder cho mỗi categorical column)
        - Feature columns list
        - Visualizations
        - Reports
        """
        
        save_artifacts()
        
        # ============================================================
        # BƯỚC 10: END MLFLOW RUN & SYNC TO S3
        # ============================================================
        
        current_run_id = mlflow. GET_ACTIVE_RUN_ID()
        
        mlflow_manager.END_RUN(status="SUCCESS")
        
        # Sync artifacts to S3 for persistence
        LOG("Syncing artifacts to S3...")
        
        TRY:
            s3_manager = TẠO_MLflowS3Manager()
            s3_manager.SYNC_ARTIFACTS_TO_S3(current_run_id)
            
            LOG("✓ Successfully synced to S3")
            
            # Verify S3 artifacts
            verification = VERIFY_S3_ARTIFACTS(
                run_id=current_run_id,
                expected_artifacts=[
                    'models/', 
                    'scalers. pkl', 
                    'encoders. pkl', 
                    'feature_cols.pkl',
                    'visualizations/',
                    'reports/'
                ]
            )
            
            LOG_VERIFICATION_RESULTS(verification)
            
            NẾU verification["success"] == False:
                LOG_WARNING("S3 verification failed!")
            
        EXCEPT Exception e: 
            LOG_ERROR(f"S3 sync failed: {e}")
        END TRY
        
    EXCEPT Exception e:
        mlflow_manager.END_RUN(status="FAILED")
        THROW e
    END TRY
    
    RETURN results
END FUNCTION
```

---

## 10. SAVE ARTIFACTS

```pseudocode
FUNCTION save_artifacts():
    """
    Mục đích: Save tất cả artifacts cần thiết cho deployment
    
    Artifacts: 
    - scalers.pkl: StandardScaler object
    - encoders.pkl: Dictionary of LabelEncoders
    - feature_cols.pkl: List of feature column names
    - models/: Folder chứa tất cả models
    
    Tại sao cần save? 
    - Deployment:  Load lại để predict trên production
    - Reproducibility: Đảm bảo transform data giống nhau
    - Consistency: Cùng preprocessing pipeline
    """
    
    # Save scalers
    JOBLIB_DUMP(scalers, '/tmp/scalers.pkl')
    
    # Save encoders
    JOBLIB_DUMP(encoders, '/tmp/encoders.pkl')
    
    # Save feature columns
    JOBLIB_DUMP(feature_cols, '/tmp/feature_cols.pkl')
    
    # Create model directories
    TẠO_FOLDER('/tmp/models/xgboost')
    TẠO_FOLDER('/tmp/models/lightgbm')
    TẠO_FOLDER('/tmp/models/ensemble')
    
    # Save individual models
    NẾU 'xgboost' TRONG models: 
        JOBLIB_DUMP(models['xgboost'], 
                   '/tmp/models/xgboost/xgboost_model.pkl')
    
    NẾU 'lightgbm' TRONG models: 
        JOBLIB_DUMP(models['lightgbm'], 
                   '/tmp/models/lightgbm/lightgbm_model.pkl')
    
    NẾU 'ensemble' TRONG models:
        JOBLIB_DUMP(models['ensemble'], 
                   '/tmp/models/ensemble/ensemble_model.pkl')
    
    # Log tất cả artifacts vào MLflow
    mlflow_manager.LOG_ARTIFACTS('/tmp/')
    
    LOG("✓ Artifacts saved successfully")
END FUNCTION
```

---

## 11. VISUALIZATIONS

```pseudocode
FUNCTION generate_and_log_visualizations(results, test_df, target_col):
    """
    Mục đích:  Tạo comprehensive visualizations cho model comparison
    
    Visualizations: 
    1. Metrics Comparison - So sánh RMSE, MAE, MAPE, R² giữa models
    2. Predictions vs Actual - Timeline chart
    3. Residuals Analysis - Scatter plot của errors
    4. Error Distribution - Histogram của residuals
    5. Feature Importance - Bar charts cho tree models
    6. Summary Statistics - Table tổng hợp
    """
    
    LOG("Starting visualization generation...")
    
    visualizer = TẠO_ModelVisualizer()
    
    # ============================================================
    # CHUẨN BỊ DATA
    # ============================================================
    
    # Extract metrics từ results
    metrics_dict = {}
    CHO MỖI (model_name, model_results) TRONG results:
        NẾU 'metrics' TRONG model_results:
            metrics_dict[model_name] = model_results['metrics']
    
    # Prepare predictions data
    predictions_dict = {}
    CHO MỖI (model_name, model_results) TRONG results:
        NẾU 'predictions' TRONG model_results: 
            pred_df = test_df[['date']].COPY()
            pred_df['prediction'] = model_results['predictions']
            predictions_dict[model_name] = pred_df
    
    # Extract feature importance
    feature_importance_dict = {}
    CHO MỖI (model_name, model_results) TRONG results:
        NẾU model_name TRONG ['xgboost', 'lightgbm']:
            model = model_results['model']
            NẾU model CÓ_ATTRIBUTE 'feature_importances_': 
                importance_df = TẠO_DATAFRAME({
                    'feature': feature_cols,
                    'importance': model. feature_importances_
                })
                importance_df = SẮP_XẾP(importance_df, 'importance', GIẢM_DẦN=True)
                feature_importance_dict[model_name] = importance_df
    
    # ============================================================
    # GENERATE VISUALIZATIONS
    # ============================================================
    
    TẠO_TEMP_DIRECTORY() AS temp_dir:
        
        LOG(f"Creating visualizations in:  {temp_dir}")
        
        # Generate all visualizations
        saved_files = visualizer.CREATE_COMPREHENSIVE_REPORT(
            metrics_dict=metrics_dict,
            predictions_dict=predictions_dict,
            actual_data=test_df,
            feature_importance_dict=feature_importance_dict,
            save_dir=temp_dir
        )
        
        LOG(f"Generated {ĐẾM(saved_files)} files")
        
        # ============================================================
        # LOG TO MLFLOW
        # ============================================================
        
        CHO MỖI (viz_name, file_path) TRONG saved_files:
            NẾU FILE_TỒN_TẠI(file_path):
                mlflow. LOG_ARTIFACT(file_path, "visualizations")
                LOG(f"Logged:  {viz_name}")
            NGƯỢC_LẠI: 
                LOG_WARNING(f"File not found: {file_path}")
        
        # ============================================================
        # CREATE COMBINED HTML REPORT
        # ============================================================
        
        create_combined_html_report(saved_files, temp_dir)
        
        combined_report = f"{temp_dir}/model_comparison_report.html"
        NẾU FILE_TỒN_TẠI(combined_report):
            mlflow.LOG_ARTIFACT(combined_report, "reports")
            LOG("Logged combined HTML report")
    
    END TEMP_DIRECTORY
END FUNCTION


FUNCTION create_combined_html_report(saved_files, save_dir):
    """
    Mục đích:  Tạo single HTML page chứa tất cả visualizations
    
    Output:  Interactive HTML report dễ share & view
    """
    
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body { 
                font-family: Arial; 
                margin: 20px; 
                background:  #f5f5f5; 
            }
            .section {
                background: white;
                padding: 20px;
                margin-bottom: 20px;
                border-radius: 8px;
                box-shadow:  0 2px 4px rgba(0,0,0,0.1);
            }
            h1, h2 { color: #333; }
            img { 
                max-width: 100%; 
                height: auto; 
                border-radius: 4px; 
            }
        </style>
    </head>
    <body>
        <h1>Sales Forecast Model Comparison Report</h1>
        <p>Generated:  {TIMESTAMP()}</p>
    """
    
    # Define sections
    sections = [
        ('metrics_comparison', 'Model Performance Metrics'),
        ('predictions_comparison', 'Predictions Comparison'),
        ('residuals_analysis', 'Residuals Analysis'),
        ('error_distribution', 'Error Distribution'),
        ('feature_importance', 'Feature Importance'),
        ('summary', 'Summary Statistics')
    ]
    
    # Add each visualization
    CHO MỖI (key, title) TRONG sections:
        NẾU key TRONG saved_files:
            html += f'<div class="section"><h2>{title}</h2>'
            
            # Base64 encode image
            img_data = BASE64_ENCODE(ĐỌC_FILE(saved_files[key]))
            html += f'<img src="data:image/png;base64,{img_data}">'
            
            html += '</div>'
    
    html += """
    </body>
    </html>
    """
    
    # Save combined report
    GHI_FILE(f"{save_dir}/model_comparison_report.html", html)
END FUNCTION
```

---

## 12. USAGE EXAMPLE

```pseudocode
# ============================================================
# MAIN USAGE WORKFLOW
# ============================================================

# 1. INITIALIZE TRAINER
trainer = ModelTrainer(config_path="/path/to/ml_config.yaml")

# 2. LOAD DATA
df = ĐỌC_CSV("sales_data.csv")

# 3. PREPARE DATA (split chronologically)
train_df, val_df, test_df = trainer.prepare_data(
    df=df,
    target_col='sales',
    date_col='date',
    group_cols=['store_id'],
    categorical_cols=['store_type', 'city']
)

# 4. TRAIN ALL MODELS
results = trainer.train_all_models(
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    target_col='sales',
    use_optuna=True  # Enable hyperparameter tuning
)

# 5. RESULTS
"""
results = {
    'xgboost': {
        'model': <XGBRegressor object>,
        'metrics': {'rmse': 1250.5, 'mae': 890.2, 'mape': 12.3, 'r2':  0.87},
        'predictions': array([... ])
    },
    'lightgbm': {
        'model': <LGBMRegressor object>,
        'metrics':  {'rmse': 1220.3, 'mae': 875.1, 'mape': 11.9, 'r2': 0.88},
        'predictions': array([...])
    },
    'prophet': {
        'model': <Prophet object>,
        'metrics': {'rmse': 1350.8, 'mae': 950.3, 'mape': 13.5, 'r2': 0.84},
        'predictions': array([...])
    },
    'ensemble': {
        'model':  <EnsembleModel object>,
        'metrics': {'rmse': 1200.1, 'mae': 860.5, 'mape': 11.5, 'r2': 0.89},
        'predictions': array([...])
    }
}
"""

# 6. BEST MODEL
best_model_name = MIN(results, KEY=lambda x: results[x]['metrics']['rmse'])
best_model = results[best_model_name]['model']

LOG(f"Best model: {best_model_name}")
LOG(f"RMSE: {results[best_model_name]['metrics']['rmse']}")

# 7. PREDICT NEW DATA
new_predictions = best_model. PREDICT(new_data)
```

---

## 13. KEY CONCEPTS SUMMARY

### A. TIME SERIES BEST PRACTICES

```
1. CHRONOLOGICAL SPLIT (không random!)
   - Train: Data cũ nhất
   - Val: Data gần đây
   - Test: Data mới nhất
   
2. FEATURE ENGINEERING
   - Lag features:  Giá trị quá khứ
   - Rolling features: Moving averages
   - Seasonal features:  Yearly/monthly patterns
   
3. AVOID DATA LEAKAGE
   - Không dùng future information
   - Fit trên train, transform trên val/test
   - Cẩn thận với target encoding
```

### B. ENSEMBLE STRATEGIES

```
1. SIMPLE AVERAGE
   pred_ensemble = (pred1 + pred2 + pred3) / 3
   
2. WEIGHTED AVERAGE
   pred_ensemble = w1*pred1 + w2*pred2 + w3*pred3
   Weights based on validation performance
   
3. STACKING
   Train meta-model trên predictions của base models
```

### C. HYPERPARAMETER TUNING WITH OPTUNA

```
1. DEFINE SEARCH SPACE
   - Ranges cho mỗi hyperparameter
   - Log scale cho learning rate
   
2. OBJECTIVE FUNCTION
   - Train model với suggested params
   - Return metric cần optimize (RMSE)
   
3. OPTIMIZATION
   - Bayesian optimization (smart search)
   - Early stopping bad trials
   - Find best params in n_trials
```

### D. MLFLOW TRACKING

```
1. LOG PARAMETERS
   - Data sizes, n_features
   - Model hyperparameters
   
2. LOG METRICS
   - Training metrics:  RMSE, MAE, R²
   - Per-model và ensemble metrics
   
3. LOG ARTIFACTS
   - Models (. pkl files)
   - Scalers, encoders
   - Visualizations
   - Reports
   
4. COMPARE RUNS
   - MLflow UI để compare experiments
   - Find best run
```

---

## 14. COMMON ISSUES & SOLUTIONS

```
ISSUE:  Overfitting (train R² = 0.95, test R² = 0.70)
SOLUTION:
- Reduce model complexity (max_depth, num_leaves)
- Increase regularization (reg_alpha, reg_lambda)
- More data hoặc data augmentation
- Cross-validation

ISSUE: Underfitting (train R² = 0.60, test R² = 0.58)
SOLUTION:
- Increase model complexity
- More features
- Feature engineering
- Longer training (more estimators)

ISSUE: High MAPE nhưng low RMSE
SOLUTION:
- MAPE sensitive với small values
- Check distribution của target
- Consider log transform

ISSUE: Prophet fails / memory issues
SOLUTION:
- Reduce regressors
- Disable MCMC
- Simplify seasonality
- Use minimal config

ISSUE: Long training time
SOLUTION:
- Reduce n_estimators
- Reduce optuna_trials
- Use early stopping
- Parallelize với n_jobs
```

---

## 15. DEPLOYMENT WORKFLOW

```
1. LOAD ARTIFACTS
   scalers = joblib.load('scalers.pkl')
   encoders = joblib.load('encoders.pkl')
   feature_cols = joblib.load('feature_cols.pkl')
   model = joblib.load('models/ensemble/ensemble_model.pkl')

2. PREPROCESS NEW DATA
   new_df = create_features(raw_data)
   new_df = encode_categorical(new_df, encoders)
   new_df = scale_features(new_df, scalers)
   X_new = new_df[feature_cols]

3. PREDICT
   predictions = model.predict(X_new)

4. POST-PROCESS
   # Inverse transform nếu cần
   predictions = inverse_log_transform(predictions)
   
   # Thêm business logic
   predictions = apply_business_rules(predictions)

5. RETURN RESULTS
   return predictions
```

---

## END OF PSEUDOCODE