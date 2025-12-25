import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
import time
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from typing import Optional, List, Union, Dict, Literal
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
import optuna

from include.feature_pipeline import FeatureEngineer
from include.utils.mlflow_utils import MLFlowManager
from include.utils.s3_utils import S3Manager
from include.ml_models.ensemble_model import EnsembleModel
from include.evaluate import diagnose_model_performance

class ModelTrainer: 
  def __init__(self, app_config: dict):
    self.model_config = app_config['models']
    self.training_config = app_config['training']
    self.data_config = app_config['dataset']

    self.feature_engineer = FeatureEngineer(app_config=app_config)
    self.mlflow_manager = MLFlowManager(app_config=app_config)
    self.s3_manager = S3Manager(app_config)
    # self.evaluator = Evaluator(app_config=app_config)

    self.encoders = {}
    self.scalers = {}
    self.models = {}

  def prepare_data(self, 
                   df: pd.DataFrame, 
                   target_col: str,
                   date_col: str, 
                   group_cols: Optional[List[str]]=None,
                   categorical_cols: Optional[List[str]]=None
                   ) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    1. Kiểm tra các cột còn thiếu từ required_cols và cols từ df 
    2. Tạo dữ liệu feature từ feature pipline 
    3. Split features thành train/val/test 
    """
    # 1. Kiểm tra missing  cols
    required_cols = [target_col, date_col]
    if group_cols: 
      required_cols.extend(group_cols)
    
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols: 
      raise ValueError(f"Missing cols: {missing_cols}")

    # 2. Tạo dữ liệu features cho model
    sales_features = self.feature_engineer.create_all_features(df=df, 
                                                               target_col=target_col, 
                                                               date_col=date_col, 
                                                               group_cols=group_cols, 
                                                               categorical_cols=categorical_cols)
    
    # 3. Split data
    # sắp xếp theo thời gian giảm dần (time series)
    sales_features = sales_features.sort_values(by=date_col, ascending=True)

    total_size = len(sales_features)
    train_size = int(total_size * self.training_config['train_size'])
    val_size = int(total_size * self.training_config['val_size'])

    # Dùng dữ liệu cũ nhất để train và mới nhất để test/val => phản ánh xu hướng hiện tại, tránh overfit và pattern cũ 
    train_df = sales_features[:train_size] 
    val_df = sales_features[train_size: train_size + val_size] 
    test_df = sales_features[train_size + val_size: ] 

    logger.info(f"Created data split: Train - {len(train_df)} Val - {len(val_df)} Test - {len(test_df)}")

    return train_df, val_df, test_df

  def preprocess_features(self, 
                             train_df: pd.DataFrame, 
                             val_df: pd.DataFrame, 
                             test_df: pd.DataFrame, 
                             target_col: str, 
                             exclude_cols: List[str]=None): 
    """
    Format dữ liệu vào ml models
      - Tách X(features) và Y(labels)
      - Encode categorical -> number 
      - Scale number 
    Args: 
      train/val/test_df 
    Output: 
      X_train/X_val/X_test, Y_train/Y_val/Y_test
    """

    # 1. Separate features và labels
    exclude_cols = [target_col] + exclude_cols if exclude_cols else [target_col]
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]

    X_train = train_df[feature_cols].copy()
    X_val = val_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()

    y_train = train_df[target_col].to_numpy()
    y_val = val_df[target_col].to_numpy()
    y_test = test_df[target_col].to_numpy()

    # 2. Encode categorical labels
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols: 
      if col not in self.encoders: 
        self.encoders[col] = LabelEncoder()
        X_train.loc[:, col] = self.encoders[col].fit_transform(X_train[col])
      else: 
        X_train.loc[:, col] = self.encoders[col].transform(X_train[col])
      
      X_val.loc[:, col] = self.encoders[col].transform(X_val[col])
      X_test.loc[:, col] = self.encoders[col].transform(X_test[col])
    
    # 3. Scale numerical features
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    x_val_scaled = scaler.transform(X_val)
    x_test_scaled = scaler.transform(X_test)

    self.scalers['standard'] = scaler
    self.feature_cols = feature_cols

    X_train_scaled = pd.DataFrame(data=x_train_scaled, columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(data=x_val_scaled, columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(data=x_test_scaled, columns=feature_cols, index=X_test.index)

    return (
      X_train_scaled, 
      X_val_scaled, 
      X_test_scaled, 
      y_train, 
      y_val, 
      y_test
    )
  
  def calculate_metrics(self, y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    metrics = {
      "rmse": np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true)), 
      "mae": mean_absolute_error(y_true=y_true, y_pred=y_pred), 
      "mape": np.mean(np.abs(y_true-y_pred)) * 100, 
      "r2": r2_score(y_pred=y_pred, y_true=y_true)
    }
    return metrics

  def train_xgboot_model(self, 
                         X_train: pd.DataFrame, 
                         X_val: pd.DataFrame, 
                         y_train: np.ndarray, 
                         y_val: np.ndarray, 
                         use_optuna: bool=True): 
    
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

    logger.info("Start training xgboost model")
    if use_optuna: 
      def objective(trial): 
        """
        Mục đích:  Function Optuna sẽ tối ưu
            
        Input:  trial - một lần thử với một set params
        Output: metric cần minimize (RMSE)
        """
        params = {
          "n_estimators": trial.suggest_int("n_estimators", 50, 300), # số trees - nhiều = phức tạp hơn và rủi ro overfit
          "max_depth": trial.suggest_int("max_depth", 3, 10), # độ sâu của mỗi tree - sâu = capture details hơn 
          "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3), # tốc độ học - thấp = chậm nhưng stable
          "subsample": trial.suggest_float("subsample", 0.6, 1.0) , # % samples dùng cho mỗi tree - <1 giúp ngăn overfit
          "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), # % feeatures dùng cho mỗi tree 
          "gamma": trial.suggest_float("gamma", 0, 0.5), #  minimum loss reduration - cao = thận trọng 
          "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0), # L1 regularization
          "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0), # L2 regularization
          "random_state": 42 # for reproducibility
        }
        # khởi tạo model và train

        params['early_stopping_rounds']=self.training_config['early_stop']
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)])
        # predict và tính rmse
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        return rmse

      study = optuna.create_study(
        direction="minimize", 
        sampler=optuna.samplers.TPESampler(seed=42), 
        pruner=optuna.pruners.MedianPruner()
      )

      # chạy optimization 
      n_trials = self.training_config['optuna_trials']
      study.optimize(func=objective, n_trials=n_trials)
      best_params = study.best_params
      best_params['random_state'] = 42
    
    else: 
      best_params = self.model_config['xgboost']['params']

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              verbose=True)

    self.models['xgboost'] = model
    return model
  
  def train_lightgbm_model(self, 
                           X_train: pd.DataFrame, 
                           X_val: pd.DataFrame, 
                           y_train: np.ndarray, 
                           y_val: np.ndarray, 
                           use_optuna: bool=True): 
    """
    Mục đích: Train LightGBM model
    
    LightGBM vs XGBoost:
    - Cũng là Gradient Boosting nhưng faster
    - Leaf-wise growth (vs level-wise của XGBoost)
    - Better với large datasets
    - Thường cho kết quả tương đương hoặc tốt hơn XGBoost
    
    Logic tương tự XGBoost
    """
    logger.info("Start training lightgbm model")

    if use_optuna: 
      def objective(trial): 
        params = {
          "num_leaves": trial.suggest_int("num_leaves", 20, 100), # Số leaves
          "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
          "n_estimators": trial.suggest_int("n_estimators", 50, 300), 
          "min_child_samples": trial.suggest_int("min_child_samples", 10, 50), # minimum samples trong leaf
          "subsample": trial.suggest_float("subsample", 0.6, 1.0), 
          "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0), 
          "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0), # L1 regularization
          "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0), # L2 regularization
          'random_state': 42,
          'verbosity': -1,
          'objective': 'regression',
          'metric': 'rmse',
          'boosting_type':  'gbdt'
        }

        # khởi tạo và train model 
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_true=y_val, y_pred=y_pred))
        return rmse

      study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
      )
      n_trials = self.training_config['optuna_trials']
      study.optimize(func=objective, n_trials=n_trials)

      best_params = study.best_params
      best_params['random_state'] = 42
      best_params['verbosity'] = -1
    
    else: 
      best_params = self.model_config['lightgbm']['params']
    
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train, 
              eval_set=[(X_val, y_val)], 
              callbacks=[lgb.early_stopping(50)])
    self.models['lightgbm'] = model
    return model

  def create_ensemble_weight(self, 
                          xgb_metrics: dict,
                          lgb_metrics: dict,
                          prophet_metrics: Optional[dict]=None) -> dict[str, float]: 
    """
    - Kết hợp predictions từ nhiều models
    - Thường tốt hơn individual models
    - Reduce variance, more robust
    
    Strategies:
    1. Simple average:  (pred1 + pred2 + pred3) / 3
    2. Weighted average: w1*pred1 + w2*pred2 + w3*pred3
    3. Stacking: Train meta-model trên predictions
    
    Dùng weighted average based on validation R²
    Ý tưởng weights:
        - Model có R² cao hơn → weight lớn hơn
        - Minimum weight = 0.2 (không ignore model hoàn toàn)
        - Normalize về tổng = 1
        
        Example:
        - xgb_r2 = 0.85, lgb_r2 = 0.80
        - xgb_weight = 0.85/(0.85+0.80) = 0.515
        - lgb_weight = 0.80/(0.85+0.80) = 0.485
    """
    r2_score_prophet = 0
    if prophet_metrics: 
      r2_score_prophet = prophet_metrics['r2']
    r2_score_xgb = xgb_metrics['r2']
    r2_score_lgb = lgb_metrics['r2']

    total_score = r2_score_prophet + r2_score_lgb + r2_score_xgb  
    xgb_weight =max(0.2, r2_score_xgb / total_score)
    lgb_weight = max(0.2, r2_score_lgb / total_score)
    if r2_score_prophet > 0:
      prophet_weight = max(0.2, r2_score_prophet / total_score)
    else: 
      prophet_weight = 0
    
    total_weight = xgb_weight + lgb_weight + prophet_weight 
    xgb_weight = xgb_weight / total_weight
    lgb_weight = lgb_weight / total_weight
    prophet_weight = prophet_weight / total_weight

    ensemble_weight = {
      "xgboost": xgb_weight, 
      "lightgbm": lgb_weight,
    }
    
    if prophet_weight > 0: 
      ensemble_weight['prophet'] = prophet_weight
    
    return ensemble_weight
  
  def train(self, 
            train_df: pd.DataFrame, 
            val_df: pd.DataFrame, 
            test_df: pd.DataFrame,
            target_col: str="sales",
            use_optuna: bool=True):
    """
    refer: docs/training.md

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
    results = {}
    # 1. 
    run_id = self.mlflow_manager.start_run(
      f"sale_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
      tags={"model_type": "ensemble", "use_optuna": str(use_optuna)}
    )
    results['run_id'] = run_id

    try: 
      # 2. process data 
      X_train, X_val, X_test, y_train, y_val, y_test = \
        self.preprocess_features(
        train_df=train_df, 
        val_df=val_df, 
        test_df=test_df, 
        target_col=target_col,
        exclude_cols=['date']
      )

      # log params data
      self.mlflow_manager.log_param(params={
        "train_size": len(X_train), 
        "val_size": len(X_val),
        "test_size": len(X_test), 
        "n_features": X_train.shape[1]
      }, run_id=run_id)
    except Exception as e: 
      logger.error(f"Error during prepare data. {str(e)}")

    try: 
      # 3. train individual model
      # 3.1 train xgboot model
      xgb_model = self.train_xgboot_model(X_train=X_train, X_val=X_val, 
                                          y_train=y_train, y_val=y_val, 
                                          use_optuna=use_optuna)
      xgb_result = xgb_model.predict(X_test)
      xgb_metrics = self.calculate_metrics(y_true=y_test, y_pred=xgb_result)

      self.mlflow_manager.log_metric(metrics={
        f"xgboost_{k}": v for k, v in xgb_metrics.items()
      }, run_id=run_id)

      self.mlflow_manager.log_model(model_name="xgboost", model=xgb_model, run_id=run_id)

      feature_importance = pd.DataFrame({
        "feature": self.feature_cols, 
        "importances": xgb_model.feature_importances_
      }).sort_values(by="importances", ascending=False).head(20)

      self.mlflow_manager.log_param({
        f"xgb_top_feature_{i}": f"feature_{row['feature']}" for i, (_, row) in enumerate(feature_importance.iterrows())
      }, run_id=run_id)

      results['xgboost'] = {
        "model": xgb_model, 
        "metrics": xgb_metrics, 
        "predictions": xgb_result
      }
    except Exception as e: 
      logger.error(f"Error during train xgboost model. {str(e)}")

    try: 
      # 3.2 train lightgbm
      lgb_model = self.train_lightgbm_model(X_train=X_train, X_val=X_val, 
                                            y_train=y_train, y_val=y_val, 
                                            use_optuna=use_optuna)
      lgb_result = lgb_model.predict(X_test)
      lgb_metrics = self.calculate_metrics(y_true=y_test, y_pred=lgb_result)

      self.mlflow_manager.log_metric(metrics={
        f"lightgbm_{k}": v for k, v in lgb_metrics.items()
      }, run_id=run_id)

      self.mlflow_manager.log_model(model_name="lightgbm", model=lgb_model, run_id=run_id)

      feature_importance = pd.DataFrame({
        "feature": self.feature_cols, 
        "importances": lgb_model.feature_importances_
      }).sort_values(by="importances", ascending=False).head(20)

      self.mlflow_manager.log_param({
        f"lgb_top_feature_{i}": f"feature_{row['feature']}" for i, (_, row) in enumerate(feature_importance.iterrows())
      }, run_id=run_id)

      results['lightgbm'] = {
        "model": lgb_model, 
        "metrics": lgb_metrics, 
        "predictions": lgb_result
      }
    except Exception as e: 
      logger.error(f"Error during traing lightgbm model. {str(e)}")

    try:
      # 4. create ensemble model

      ensemble_models = {
        "xgboost": xgb_model, 
        "lightgbm": lgb_model,
      }
      ensemble_weights = self.create_ensemble_weight(xgb_metrics=xgb_metrics, 
                                                    lgb_metrics=lgb_metrics)
      
      # logger.debug(f"ensemble weights: {ensemble_weights}")
      ensemble_results = xgb_result * ensemble_weights['xgboost'] \
                      + lgb_result + ensemble_weights['lightgbm']
      # logger.debug(f"ensemble results: {ensemble_results}")

      ensemble_model = EnsembleModel(ensemble_models=ensemble_models, 
                                     ensemble_weights=ensemble_weights)
      
      ensemble_metrics = self.calculate_metrics(y_pred=ensemble_results, y_true=y_test)
      # logger.debug(f"ensemble metrics: {ensemble_metrics}")

      self.mlflow_manager.log_model(model_name="ensemble", model=ensemble_model, run_id=run_id)
      self.mlflow_manager.log_metric({
        f"ensemble_{key}": value for key, value in ensemble_metrics.items() 
      }, run_id=run_id)

      self.models['ensemble'] = ensemble_model
      results['ensemble'] = {
        "model": ensemble_model, 
        "metrics": ensemble_metrics, 
        "predictions": ensemble_results
      }
    except Exception as e: 
      logger.error(f"Error during create ensemble model. {str(e)}")
    
    try:
      # 5. Run diagnosis 
      test_pred = {
        "lightgbm": lgb_result, 
        "xgboost": xgb_result, 
        "ensemble": ensemble_results
      }

      import json
      test_pred_json = json.dumps({k: v.tolist() for k, v in test_pred.items()})
      with open('test_predictions.json', 'w') as f:
        f.write(test_pred_json)
      logger.info("Saved test predictions to JSON")
      
      diagnosis = diagnose_model_performance(train_df=train_df, 
                                             val_df=val_df, 
                                             test_df=test_df, 
                                             predictions=test_pred, 
                                             target_col=target_col)
      
      logger.info("Diagnostic recommendations:")
      for rec in diagnosis['recommendations']:
        logger.warning(f"- {rec}")

      # 6. Lưu artifacts vào s3 
      dst_path = self.data_config['artifact_path']
      bucket_name = self.data_config['artifact_bucket']
      artifact_dirs = self.mlflow_manager.download_artifacts_from_s3(run_id=run_id, dst_path=dst_path)
      dt = datetime.fromtimestamp(time.time())
      current_time = dt.strftime("%d-%m-%Y")

      try: 
        for root, dirs, files in os.walk(artifact_dirs): 
          for file in files: 
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, artifact_dirs)
            s3_key = f"{current_time}/{run_id}/artifacts/{relative_path}"
            self.s3_manager.upload_file(file_path=local_path, bucket_name=bucket_name, s3_key=s3_key)
        logger.info(f"Upload mlflow artifacts to s3 sucessfully!")
      except Exception as e: 
        logger.error(f"Failed upload mlflow artifacts to s3. {str(e)}")


      # 7. Lưu encoder, scalers, feature_cols vào mlflow 
      self.save_artifacts(run_id=run_id)
      logger.info("Save preprocesor (encoder, scaler, feature_cols) to mlflow sucessfully")

      return results

    except Exception as e: 
      logger.error(f"Error during process callback. {str(e)}")

  def save_artifacts(self, run_id: str): 
    import joblib
    dst_path = os.path.join(self.data_config['artifact_path'], "preprocess") 
    
    os.makedirs(dst_path, exist_ok=True)

    joblib.dump(self.scalers, os.path.join(dst_path, "scalers.pkl"))
    joblib.dump(self.encoders, os.path.join(dst_path, "encoders.pkl"))
    joblib.dump(self.feature_cols, os.path.join(dst_path, "feature_cols.pkl"))

    self.mlflow_manager.log_artifacts(run_id=run_id, artifact_path=dst_path)
  

if __name__ == "__main__": 

  from dotenv import load_dotenv
  load_dotenv('.env.dev')

  from include.utils.helpers import load_config
  app_config_path = "/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml"
  processed_csv_path = "/home/ducpham/workspace/Sales-Forecasting-Mlops/data/csv/processed.csv"
  
  app_config = load_config(app_config_path)
  sale_processed = pd.read_csv(processed_csv_path)

  trainer = ModelTrainer(app_config=app_config)
  train_df, val_df, test_df = trainer.prepare_data(df=sale_processed, 
                                                  date_col="date",
                                                  target_col="sales", 
                                                  group_cols=["store_id"],
                                                  categorical_cols=None)
  

  
  X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test = \
      trainer.preprocess_features(train_df=train_df, 
                                  val_df=val_df, 
                                  test_df=test_df, 
                                  target_col="sales", 
                                  exclude_cols=['date'])

  # model = trainer.train_lightgbm_model(X_train=X_train_scaled, 
  #                              X_val=X_val_scaled, 
  #                              y_train=y_train, 
  #                              y_val=y_val, 
  #                              use_optuna=True)
  
  result = trainer.train(train_df=train_df, val_df=val_df, test_df=test_df, 
                        target_col="sales", use_optuna=True)

  print(result)



