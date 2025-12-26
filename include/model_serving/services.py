import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
import joblib
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, Any
from datetime import datetime
from loguru import logger
from utils import load_config, MLFlowManager, S3Manager
from data_loader import DataLoader
from feature_pipeline import FeatureEngineer
from models import RequestModel, ResponseModel


# config = load_config('/usr/local/airflow/include/config.yaml')
config = load_config('/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml')


class ModelInferenceService: 
  def __init__(self): 
    self.models = {}
    self.encoders = {}
    self.scalers = {}
    self.feature_cols = None
    self.model_version = None

    self.mlflow_config = config['mlflow']
    self.data_config = config['dataset']
    self.mlflow_manager = MLFlowManager(app_config=config)
    self.feature_engineer = FeatureEngineer(app_config=config)
    self.data_loader = DataLoader(app_config=config)


  def load_models(self, stage: str="Production"):
    """
    1. Tải và load model từ artifacts 
    2. Tải và load scalers, encoders, feature_cols từ artifacts
    """

    try: 
      # load models
      best_model_version = self.mlflow_manager.get_best_run()
      # experiment_id = best_model_version['experiment_id']
      run_id = best_model_version['run_id']
      
      for model in ["xgboost", "lightgbm", "ensemble"]:
        # Chỉ cần relative path trong artifacts, không phải full URI
        artifact_path = f"models/{model}"
        dst_path = os.path.join(self.data_config['artifact_path'], "models")
        
        # Load model từ file đã download
        model_file = os.path.join(dst_path, model, f"{model}_model.pkl")
        if os.path.exists(model_file):
          self.models[model] = joblib.load(model_file)
          logger.info(f"Loaded {model} successfully")
        else:
          logger.info(f"Downloading {model} from run {run_id}...")
        self.mlflow_manager.download_artifacts_from_s3(run_id=run_id, path=artifact_path, dst_path=dst_path)

      # Load preprocessing artifacts
      preprocessor_artifacts = {
        "encoders": "encoders.pkl",
        "scalers": "scalers.pkl", 
        "feature_cols": "feature_cols.pkl"
      }
      
      for artifact_name, filename in preprocessor_artifacts.items():
        artifact_path = f"{filename}"
        dst_path = os.path.join(self.data_config['artifact_path'], "preprocessor")
        
        # Load artifact
        file_path = os.path.join(dst_path, filename)
        if os.path.exists(file_path):
          if artifact_name == "encoders":
            self.encoders = joblib.load(file_path)
          elif artifact_name == "scalers":
            self.scalers = joblib.load(file_path)
          elif artifact_name == "feature_cols":
            self.feature_cols = joblib.load(file_path)
          logger.info(f"Loaded {artifact_name} successfully")
        else:
          logger.info(f"Downloading {artifact_name}...")
        self.mlflow_manager.download_artifacts_from_s3(run_id=run_id, path=artifact_path, dst_path=dst_path)

    except Exception as e: 
      logger.error(f"Error during load models. {str(e)}")

  def prepare_feature(self, df: pd.DataFrame, is_single_prediction: bool=False) -> pd.DataFrame:
    df_features = self.feature_engineer.create_all_features(
      df=df, 
      target_col="sales", 
      date_col="date", 
      group_cols=["store_id"], 
      categorical_cols=None
    )
    # df_features.to_csv("df_feature.csv")

    # Nếu là single prediction, chỉ lấy row cuối cùng (prediction point)
    if is_single_prediction:
      df_features = df_features.tail(1)
    
    df_features = df_features[self.feature_cols] # chỉ lấy các cột dùng để training

    # Chọn các cột object và encode 
    for col in df_features.select_dtypes(include=['object']).columns: 
      df_features[col] = self.encoders[col].transform(df_features[col].astype(str))

    # Scale feature cols 
    scaled_array = self.scalers['standard'].transform(df_features)

    df_scaled = pd.DataFrame(
        scaled_array, 
        columns=df_features.columns,  # ← Giữ nguyên feature names
        index=df_features.index
    )
    return df_scaled

  def _generate_confidence_intervals(self, 
                                    predictions: np.ndarray, 
                                    confidence_levels: list[float]=[0.8, 0.95]) -> dict: 
    """
    Mục tiêu: tạo ra 1 khoảng(interval) cho giá trị dự đoán. Để đảm bảo rằng giá trị thực sự nằm trong 1 vùng [a, b]
      - Tính 1 khoảng lề margin = Z_score * std => [a, b] = [value - margin, value + margin]
      - Z_score = 1.96 nếu confidence = 0.95, Z_score = 1.28 nếu confidence = 0.8  
    """
    intervals = defaultdict(dict)

    for model, prediction in predictions.items(): 
      for confidence in confidence_levels:
        std_dev = prediction * 0.1
        Z_score = 1.96 if confidence == 0.95 else 1.28 
        # tính margin
        margin = Z_score * std_dev

        intervals[f"{model}"].update({
          f"confidence_{confidence}": [round(prediction-margin, 2), round(prediction+margin, 2)]
        })

    return intervals

  def _get_historical_data(self, store_id: str, end_date: str, days: int=30) -> pd.DataFrame:
    data_info = self.data_loader.extract_data()
    sales_data = self.data_loader.transform_data_task(data_info=data_info)

    sales_data = sales_data[sales_data['store_id'] == store_id].sort_values('date', ascending=False).head(days)
    return sales_data

  def predict_single(self, 
                     store_id: str, 
                     date: str, 
                     additional_features: Optional[Dict[str, Any]]=None): 
    """
    Predict với historical data để tính lag/rolling features
    """
    # 1. Load historical data (30 ngày trước)
    historical_df = self._get_historical_data(
      store_id=store_id,
      end_date=date,
      days=30
    )
    
    # 2. Thêm prediction point
    prediction_data = {
      "store_id": store_id, 
      "date": pd.to_datetime(date), 
      "sales": 0  # dummy value cho feature engineering
    }
    
    if additional_features: 
      for key, value in additional_features.items(): 
        prediction_data[key] = value
    
    prediction_df = pd.DataFrame([prediction_data])
    
    # 3. Concat historical + prediction point
    df_combined = pd.concat([historical_df, prediction_df], ignore_index=True)
    # df_combined.to_csv("df_combined.csv")
    
    # 4. Feature engineering trên toàn bộ data, nhưng chỉ predict điểm cuối
    X = self.prepare_feature(df=df_combined, is_single_prediction=True)
    # 5. Predict
    xgboost_pred = self.models['xgboost'].predict(X)
    lgb_pred = self.models['lightgbm'].predict(X)
    ensemble_pred = self.models['ensemble'].predict(X)

    predictions = {
      "xgboost": round(float(xgboost_pred[0]), 2), 
      "lgb_pred": round(float(lgb_pred[0]), 2), 
      "ensemble_pred": round(float(ensemble_pred[0]), 2)
    }

    intervals = self._generate_confidence_intervals(predictions=predictions)

    return ResponseModel(
      store_id=store_id, 
      date=date, 
      predictions=predictions, 
      intervals=intervals,
      model_version="v1", 
      prediction_timestamp=datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
    )


if __name__ == "__main__": 
  from dotenv import load_dotenv
  load_dotenv(".env.dev")

  service = ModelInferenceService()
  service.load_models()

  predictions = service.predict_single(store_id="store_001", 
                                       date="2026-01-01", 
                                       additional_features={"has_promotion": 1})
  
  print(f"Predicted sales: {predictions}")