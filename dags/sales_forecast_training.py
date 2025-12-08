import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
import pandas as pd
from io import BytesIO
from loguru import logger
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from airflow.providers.standard.operators.bash import BashOperator
from include.data_generator import RealisticSalesDataGenerator
from include.utils.helpers import load_config
from include.utils.s3_utils import S3Manager
from include.ml_models import ModelTrainer

config = load_config('/usr/local/airflow/include/config.yaml')
s3_manager = S3Manager(config_app=config)


default_args = {
  "owner": "Jiyuu",
  "depends_on_past": False,
  "start_date": datetime(2025, 12, 5),
  "email": ["duc78240@gmail.com"],
  "email_on_failure": False,
  "email_on_retry": False,
  "retries": 1,
  "retry_delay": timedelta(minutes=5),
}

@dag(
  dag_id="sales_forecast_training",
  schedule="@weekly",
  start_date=datetime(2025, 12, 5),
  catchup=False,
  default_args=default_args,
  description="DAG for training sales forecasting model",
  tags=["sales_forecasting", "training"],
)
def sales_forecast_training():

  @task()
  def extract_data_task():

    config_data = config['dataset']
    output_data_path = config_data['data_path']
    bucket_name = config_data['data_bucket']

    generator = RealisticSalesDataGenerator(config_app=config) 
       
    if s3_manager.validate_object_in_bucket(bucket_name=bucket_name):
      data_path_files = s3_manager.get_files(bucket_name=bucket_name, include_url=True)
    else: 
      data_path_files = generator.generate_sales_data(output_dir=output_data_path)

    total_files = sum(len(list_files) for _, list_files in data_path_files.items())
    logger.info(f"Total data files available: {total_files}")
    for data_type, files in data_path_files.items():
      logger.info(f"{data_type} files: {len(files)}")
    
    return {
      "file_paths": data_path_files, 
      "total_files": total_files, 
      "data_output_dir": output_data_path
    }

  @task()
  def validate_data_task(data_info: dict): 

    config_data = config['dataset']
    bucket_name = config_data['data_bucket']

    total_rows = 0
    issue_found = []
    files_path = data_info['file_paths']
    for i, sale_path in enumerate(files_path['sales'][:10]): 
      file_name = sale_path['key']
      df = s3_manager.read_file_as_bytes(bucket_name=bucket_name, 
                                         s3_key=file_name)
      if file_name.endswith("parquet"):
        df = pd.read_parquet(BytesIO(df))
      else: 
        continue

      if i==0: 
        logger.info(f"Sales data columns: {df.columns.tolist()}")
      
      if df.empty: 
        issue_found.append(f"Empty sales data file at {sale_path['key']}")
        continue
        
      required_cols = config_data['required_cols']
      missing_cols = [col for col in required_cols if col not in df.columns]
      if missing_cols:
        issue_found.append(f"Missing columns {missing_cols} in file {sale_path['key']}")
      
      total_rows += len(df)

      if df['quantity_sold'].min() < 0: 
        issue_found.append(f"Negative quantity_sold in file {sale_path['key']}") 

      if df['revenue'].min() < 0: 
        issue_found.append(f"Negative revenue in file {sale_path['key']}")

    validate_summary = {
      "total_files_checked": 10,
      "total_rows": total_rows,
      "total_issues": len(issue_found),
      "issues_found": issue_found
    }

    if issue_found: 
      logger.warning(f"Data validation found issues: {issue_found}")
      for issue in issue_found: 
        logger.warning(f"- {issue}")
    else: 
      logger.info(f"Validation passed for {data_info['total_files']} files with {total_rows} rows.")
    return validate_summary

  @task()
  def transform_data_task(data_info: dict) -> pd.DataFrame: 
    """
    Chuẩn bị dữ liệu cho model. Biến đổi từ dữ liệu thô (product-level) sang dữ liệu tổng hợp (store-level).
    1. Group by và agg dữ liệu sales cho từng sản phẩm 
    2. Merge với promotions, tạo cột has_promotion = 1. Khi merge với sales, sản phẩm không có trong promotions thì có has_promotion = 0
    3. Group by và agg dữ liệu tổng hợp cho từng cửa hàng. Sau đo merge với dữ liệu sales
    4. Chuyển dữ liệu product-level sang dữ liệu store-level
    """

    bucket_name = config['dataset']['data_bucket']
    # 1. 
    file_paths = data_info['file_paths']
    sale_dfs = []
    for sale_path in file_paths['sales']: 
      sale_df = s3_manager.read_file_as_bytes(
        bucket_name=bucket_name, 
        s3_key=sale_path['key']
      ) # đọc dữ liệu trực tiếp từ s3 thành dạng bytes

      sale_df = pd.read_parquet(BytesIO(sale_df)) # chuyển bytes sang dataframe
      sale_dfs.append(sale_df)

    daily_sales = pd.concat(sale_dfs, ignore_index=True) # gộp tất cả các files parquet thành 1 dataframe

    daily_sales = daily_sales.groupby(['date', 'store_id', 'product_id', 'category']).agg({
      "quantity_sold": "sum",
      "revenue": "sum",
      "cost": "sum", 
      "profit": "sum",
      "discount_percent": "mean", 
      "unit_price": "mean"
    }).reset_index()	
    daily_sales = daily_sales.rename(columns={"revenue": "sales"}) # đổi tên cột revenue thành sales

    logger.info(f"Num of sales: {len(daily_sales)}")
     
    # 2. 
    if "promotions" in file_paths and file_paths['promotions']: 
      promotion_df = s3_manager.read_file_as_bytes(
        bucket_name=bucket_name, 
        s3_key=file_paths['promotions'][0]['key']
      )

      promotion_df = pd.read_parquet(BytesIO(promotion_df))
      promotion_summary = promotion_df.groupby(['date', 'product_id'])['discount_percent'].max().reset_index()
      promotion_summary['has_promotion'] = 1
      
      daily_sales = daily_sales.merge(
        promotion_summary[['date', 'product_id', 'has_promotion']], 
        on=['date', 'product_id'],
        how='left'
      )

    # 3. 
    if "customer_traffic" in file_paths and file_paths['customer_traffic']: 
      traffic_dfs = []
      for traffic_path in file_paths['customer_traffic']: 
        traffic_df = s3_manager.read_file_as_bytes(
          bucket_name=bucket_name, 
          s3_key=traffic_path['key']
        )
        traffic_df = pd.read_parquet(BytesIO(traffic_df))
        traffic_dfs.append(traffic_df)

      traffic_dfs = pd.concat(traffic_dfs, ignore_index=True)
      traffic_dfs = traffic_dfs.groupby(['date', 'store_id']).agg({
        'customer_traffic': 'sum', 
        'is_holiday': "max"
      })

      daily_sales = daily_sales.merge(
        traffic_dfs, 
        on=['date', 'store_id'], 
        how='left'
      )

    # 4. 
    daily_store_sales = daily_sales.groupby(['date', 'store_id']).agg({
      'quantity_sold': "sum", # tổng số lượng bán của store trong ngày
      "sales": "sum", # tổng doanh thu của store trong ngày
      "profit": "sum", # tổng lợi nhuận -> đánh giá hiệu quả kinh doanh
      "has_promotion": "mean", # tỉ lệ sản phẩm có khuyến mãi
      "customer_traffic": "first", # đã tổng hợp -> lấy giá trị duy nhất
      "is_holiday": "first", # binary lag -> giá trị duy nhất 
    }).reset_index()

    daily_store_sales['date'] = pd.to_datetime(daily_store_sales['date'])
    return daily_store_sales
    
  @task()
  def train_model_task(daily_store_sales: pd.DataFrame): 
    trainer = ModelTrainer()
    train_df, val_df, test_df = trainer.prepare_data(
      daily_store_sales, 
      target_col = "sales",
      date_col = "date",
      group_cols = ['store_id'], 
      categorical_cols = ['store_id'],
    )

    results = trainer.train_all_models(train_df, val_df, test_df, target_col="sales", use_optuna=True)


  # Invoke the DAG
  data_info = extract_data_task()
  validation_summary = validate_data_task(data_info=data_info)
  daily_store_sales = transform_data_task(data_info=data_info)
  results = train_model_task(daily_store_sales=daily_store_sales)


# Instantiate the DAG
sales_forecast_training_dag = sales_forecast_training()
