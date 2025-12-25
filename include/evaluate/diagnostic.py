
import numpy as np
import pandas as pd
from typing import Dict, Any

def detect_outliers(data: pd.Series): 
  Q1 = data.quantile(q=0.25)
  Q3 = data.quantile(q=0.75)
  IQR = Q3 - Q1

  lower_bound = Q1 - 1.5*IQR
  upper_bound = Q3 + 1.5*IQR

  outliers  = data[(data < lower_bound) | (data > upper_bound)]

  return {
    "count": len(outliers), 
    "percentage": len(outliers) / len(data), 
    "lower_bound": lower_bound, 
    "upper_bound": upper_bound, 
    "min_outlier": outliers.min(), 
    "max_outlier": outliers.max(),
  }

def diagnose_model_performance(train_df: pd.DataFrame, 
                               val_df: pd.DataFrame, 
                               test_df: pd.DataFrame, 
                               predictions: Dict[str, np.ndarray], 
                               target_col: str="sales") -> Dict[str, Any]: 
  """
    Diagnostics kiểm tra: 
    - Overfitting?  (train metrics >> test metrics)
    - Underfitting? (cả train và test metrics đều kém)
    - Residuals patterns?  (errors có pattern hay random?)
    - Feature issues? (multicollinearity, leakage?)
    
    Output:  Recommendations để improve model
  """

  # 1. Kiểm tra điểm ngoại lệ
  diagnosis = {
    "data_quanlity": {}, 
    "distribution_shift": {}, 
    "prediction_analysis": {}, 
    "recommendations": []
  }

  y_train = train_df[target_col]
  y_val = val_df[target_col]
  y_test = test_df[target_col]

  train_outliers = detect_outliers(data=y_train)
  val_outliers = detect_outliers(data=y_val)
  test_outliers = detect_outliers(data=y_test)

  diagnosis['data_quanlity']['train_outliers'] = train_outliers
  diagnosis['data_quanlity']['val_outliers'] = val_outliers
  diagnosis['data_quanlity']['test_outliers'] = test_outliers


  # 2. Kiểm tra phân phối giữa train và val/test data. Nếu quá lệch -> model học 1 kiểu, dự đoán trên kiểu khác -> cảnh báo
  train_mean, train_std = y_train.mean(), y_train.std()
  val_mean, val_std = y_val.mean(), y_val.std()
  test_mean, test_std = y_test.mean(), y_test.std()

  diagnosis['distribution_shift']['train_stats'] = {"train_mean": train_mean, "train_std": train_std}
  diagnosis['distribution_shift']['val_stats'] = {"val_mean": val_mean, "val_std": val_std}
  diagnosis['distribution_shift']['test_stats'] = {"test_mean": test_mean, "test_std": test_std}

  scale = max(abs(train_mean), train_std, 1e-8)
  mean_shift_val  = abs(val_mean - train_mean) / scale
  mean_shift_test = abs(test_mean - train_mean) / scale

  if mean_shift_val > 0.2: 
    diagnosis['recommendations'].append(f"Significant distribution shift in validation set (mean shift: {mean_shift_val:.1%})")
  
  if mean_shift_test > 0.2: 
    diagnosis['recommendations'].append(
        f"Significant distribution shift in test set (mean shift: {mean_shift_test:.1%})"
    )

  # 3. Kiểm tra dự đoán từ model có nhỏ/lớn quá mức?
  for model, pred in predictions.items(): 
    pred = np.array(pred)
    
    low_pred = (pred < y_test.min()*0.5).sum() # < giá trị thật*0.5
    high_pred = (pred > y_test.max()*1.5).sum() # > giá trị thật*1.5 


    residuals = high_pred - low_pred

    diagnosis['prediction_analysis'][model] = {
      "pred_mean": pred.mean(), 
      "pred_std": pred.std(), 
      "extreme_low": low_pred, 
      "extreme_high": high_pred, 
      "residual_mean": residuals.mean(), 
      "residual_std": residuals.std(),
      "mape": np.mean(np.abs(residuals / y_test)) * 100
    }

  
  # 4. Kiểm tra số lượng features. > 50 thì cảnh báo
  feature_cols = [col for col in train_df.columns if col not in [target_col, 'date']]
  diagnosis['data_quanlity']['n_features'] = len(feature_cols)

  if len(feature_cols) > 50: 
    diagnosis['recommendations'].append(
      f"High number of features ({len(feature_cols)}). Consider more aggressive feature selection."
    )
  
  # 5. Kiểm tra rò rỉ dữ liệu. Xem feature nào có độ tương quan với target > 0.95 (model học vẹt 1 feature giống target) 
  numeric_cols = train_df.select_dtypes(include=[np.number]).columns
  numeric_cols = [col for col in numeric_cols if col != 'sales']

  if len(numeric_cols) > 0: 
    correlations = train_df[numeric_cols].corrwith(train_df[target_col])

    high_corr = correlations[abs(correlations) > 0.95]

    if len(high_corr) > 0: 
      diagnosis['recommendations'].append(
        f"Potential data leakage: {len(high_corr)}. Feature have 95% correlation with target - {high_corr.to_dict()}"
      )

  # 6. Check số lượng sample để training 
  if len(train_df) <= 1000:
    diagnosis['recommendations'].append(
      f"Smale sample training set {len(train_df)}. Consider generating more data"
    ) 

  # 7. Phân tích cột target
  target_zeros = (y_train == 0).sum()
  if target_zeros > len(y_train)*0.1: 
    diagnosis['recommendations'].append(
      f"Many zeros sales in training. Consider log transformation or zero-inflated models"
    )

  return diagnosis
  

if __name__ == "__main__": 
  import json 
  from pathlib import Path

  file_path = Path("/home/ducpham/workspace/Sales-Forecasting-Mlops/include/test_predictions.json")
  with file_path.open("r", encoding="utf-8") as f:
    test_predictions = json.load(f)

  app_config_path = "/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml"
  processed_csv_path = "/home/ducpham/workspace/Sales-Forecasting-Mlops/data/csv/processed.csv"
  sale_processed = pd.read_csv(processed_csv_path)
