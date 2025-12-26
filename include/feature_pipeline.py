import os
import sys
import holidays
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
import pandas as pd
import numpy as np
from typing import List, Optional
from loguru import logger

class FeatureEngineer: 
  def __init__(self, app_config: dict): 
    self.app_config = app_config
    self.feature_config = app_config['features']
    self.validation_config = app_config['validation']
    self.vn_holidays = holidays.VN()

  
  def create_date_features(self, df: pd.DataFrame, date_col: str='date') -> pd.DataFrame:
    
    df = df.copy()
    date_features = self.feature_config['date_features']
    df[date_col] = pd.to_datetime(df[date_col])

    if 'day' in date_features: 
      df['day'] = df[date_col].dt.day
    if 'month' in date_features: 
      df['month'] =  df[date_col].dt.month
    if 'year' in date_features: 
      df['year'] = df[date_col].dt.year
    if 'dayofweek' in date_features: 
      df['dayofweek'] = df[date_col].dt.dayofweek
    if 'quarter' in date_features: 
      df['quarter'] = df[date_col].dt.quarter
    if 'weekofyear' in date_features: 
      df['weekofyear'] = df[date_col].dt.isocalendar().week
    if 'is_weekend' in date_features: 
      df['is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    if 'is_holiday' in date_features: 
      df['is_holiday'] = df[date_col].apply(lambda x: x in self.vn_holidays)

    logger.info(f"Created date features with columns: {df.columns.tolist()}")
    return df

  def create_lag_feature(self, 
                         df: pd.DataFrame, 
                         target_col: str, 
                         group_cols: Optional[List[str]]=None) -> pd.DataFrame: 
    
    df = df.copy()
    lag_features = self.feature_config['lag_features']

    for lag in lag_features: 
      if group_cols:
        df[f'sales_lag_{lag}'] = df.groupby(group_cols)[target_col].shift(periods=lag)
      else: 
        df[f'sales_lag_{lag}'] = df[target_col].shift(periods=lag)
    
    logger.info(f"Created lag features: {lag_features}")
    return df

  def create_rolling_feature(self, 
                             df: pd.DataFrame, 
                             target_col: str, 
                             group_cols: Optional[List[str]]=None):
    df = df.copy()
    rolling_windows = self.feature_config['rolling_features']['windows']
    rolling_funcs = self.feature_config['rolling_features']['funtions']

    for window in rolling_windows: # duyệt qua từng window
      for func in rolling_funcs: # với mỗi window => xử lý cho từng func
        col_name = f"sales_rolling_{window}_{func}"
        
        if group_cols:
          # trượt cửa sổ tối đa window và chỉ cần có 1 phần tử thì window được tính. Sau đó tổng hợp theo func
            # Nhóm 1 (sau khi groupby): store_id=A, product_id=X → [100, 120, 130] 
              # Dòng 1 (100): cửa sổ có 1 giá trị → mean = 100.0
              # Dòng 2 (120): cửa sổ [100, 120] (do min_periods=1) → mean = 110.0
              # Dòng 3 (130): cửa sổ [100, 120, 130] → mean = 116.67 
          df[col_name] = df.groupby(group_cols)[target_col].transform(
            func=lambda x: x.shift(1).rolling(window, min_periods=1).agg(func) 
          )
        else: 
          df[col_name] = df[target_col].transform(
            func=lambda x: x.shift(1).rolling(window, min_periods=1).agg(func)
          )
      logger.info(f"Created rolling features: {rolling_windows} - for funcs: {rolling_funcs}")
      return df

  def create_cyclical_features(self, df: pd.DataFrame, date_col: str="date") -> pd.DataFrame: 
    
    df["day_sin"] = np.sin(2 * np.pi * df[date_col].dt.day / 31)
    df["day_cos"] = np.cos(2 * np.pi * df[date_col].dt.day / 31)

    df['month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)

    df['dayofweek_sin'] = np.sin(2 * np.pi * df[date_col].dt.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df[date_col].dt.dayofweek / 7)

    logger.info(f"Created cyclical features")
    return df
  
  def create_interaction_features(self, 
                                  df: pd.DataFrame, 
                                  categorical_cols: Optional[List[str]]=None) -> pd.DataFrame: 
    df = df.copy()
    for i, col1 in enumerate(categorical_cols):
      for col2 in categorical_cols[i+1:]: 
        df[f"{col1}_{col2}_interaction"] = df[col1].astype(str) + "_" + df[col2].astype(str)
    
    return df

  def handle_missing_values(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame: 
    df = df.copy()
    columns = df.select_dtypes(np.number).columns

    # fill dữ liệu thiếu cho các cột rolling và lag bị missing khi tạo
    for col in columns: 
      if df[col].isnull().any(): 
        if "lag" in col or "rolling" in col: 
          df[col] = df[col].ffill().bfill()
    
    # fill nhưng giá NaN ở cột has_promotion = 0 (không khuyến mãi)
    df['has_promotion'] = df['has_promotion'].fillna(0).astype(int)
    
    # xóa các hàng missing ở target col
    df = df.dropna(subset=[target_col])
    return df

  def create_all_features(self, 
                          df: pd.DataFrame, 
                          target_col: str, 
                          date_col: str, 
                          group_cols: Optional[List[str]]=None, 
                          categorical_cols: Optional[List[str]]=None): 
    """
    1. Chuẩn bị dữ liệu: 
      - Sắp xếp data theo groups (thứ tự cửa hàng, sản phẩm v.v) và date(thứ tự thời gian)
      - LÝ DO: Đảm bảo tính toán lag/rolling đúng trình tự thời gian
    2. Tạo features về thời gian:
      - Thời gian cơ bản (năm, tháng, ngày, thứ)
      - Thời gian kinh doanh (quý, tuần)
      - Đặc điểm thời gian: cuối tuần?, ngày lễ?
      - MỤC ĐÍCH: Bắt pattern theo mùa, theo tuần, theo lễ
    3. Tạo features trong quá khứ 
      - Mỗi khoảng thời gian (7, 14, 21, v.v) ngày trước:
        - lấy giá trị sales của N ngày trước
        - Lưu vào cột sale_lag_n
      - MỤC ĐÍCH: Model biết "doanh số tuần trước", "tháng trước"
      - VÍ DỤ: Nếu tuần trước bán tốt → tuần này có thể tiếp tục tốt
    4. Tạo features thống kê 
      - Mỗi cửa sổ thời gian (7, 14, 21, v.v ngày)
        -  Tính toán sales của N ngày gần nhất theo: mean(xu hướng chung); std(độ biến động); min/max(biên độ dao động)
      - MỤC ĐÍCH: Đo lường xu hướng và độ ổn định gần đây
      - VÍ DỤ: Nếu std cao → bán hàng không ổn định
    5. Tạo features chu kỳ (cyclical)
      - Chuyển đổi các giá trị tuần hoàn sang dạng cos/sin
        - Tháng 1->12->1: chu kì năm
        - Ngày 1->31->1: chu kì tháng
        - Ngày 0->6->0: chu kì tuần
      -  MỤC ĐÍCH: Đo lường xu hướng và độ ổn định gần đây
      - VÍ DỤ: Nếu std cao → bán hàng không ổn định
    6. Tạo features kết hợp (interaction)
      - Nếu có categorical_cols: 
        - Tạo interaction_features:
        - Kết hợp các biến phân loại: store_id + product_id; region + category
      - MỤC ĐÍCH: Bắt pattern đặc biệt của từng tổ hợp
      - VÍ DỤ: Sản phẩm A bán tốt ở cửa hàng 1, nhưng kém ở cửa hàng 2
    7. Xử lý giá trị thiếu 
      - VỚI MỖI cột số:
        - NẾU có giá trị null:
          - NẾU là lag/rolling feature:
            → Lấy giá trị gần nhất (forward fill)
            → Nếu vẫn null, lấy giá trị sau (backward fill)
          - KHÔNG THÌ:
            → Lấy giá trị trung bình
      - LÝ DO: Lag/rolling tạo null ở đầu chuỗi thời gian
    """
    # 1. 
    if group_cols: 
      df = df.sort_values(by=group_cols + [date_col])
    else: 
      df = df.sort_values(by=date_col)
    # 2.
    df = self.create_date_features(df=df, date_col=date_col)
    # 3. 
    df = self.create_lag_feature(df=df, target_col=target_col, group_cols=group_cols)
    # 4. 
    df = self.create_rolling_feature(df=df, target_col=target_col, group_cols=group_cols)
    # 5. 
    df = self.create_cyclical_features(df=df, date_col=date_col)
    # 6. 
    if categorical_cols:
      df = self.create_interaction_features(df=df, categorical_cols=categorical_cols)
    # 7. 
    df = self.handle_missing_values(df=df, target_col=target_col)

    logger.info(f"Feature engineering complete. Total features: {len(df.columns)}")
    return df


if __name__ == "__main__": 
  from include.utils.helpers import load_config
  app_config_path = "/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml"
  processed_csv_path = "/home/ducpham/workspace/Sales-Forecasting-Mlops/data/csv/processed.csv"
  
  app_config = load_config(app_config_path)
  sale_processed = pd.read_csv(processed_csv_path)

  feature_engineer = FeatureEngineer(app_config=app_config)
  # df = feature_engineer.create_date_features(df=sale_processed, date_col='date')
  # df = feature_engineer.create_lag_feature(df=df, targer_col='sales', group_cols=None)
  # df = feature_engineer.create_rolling_feature(df=df, target_col='sales', group_cols=None)
  # df = feature_engineer.create_cyclical_features(df=df, date_col="date")
  # df = feature_engineer.create_interaction_features(df=df, categorical_cols=["date", "store_id"])
  # df = feature_engineer.handle_missing_values(df=df)

  df = feature_engineer.create_all_features(df=sale_processed, 
                                            target_col="sales", 
                                            date_col="date", 
                                            group_cols=["store_id"], 
                                            categorical_cols=None)
  df.to_csv(processed_csv_path)
  print(df.columns.tolist())