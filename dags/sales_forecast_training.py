import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
import pandas as pd
from io import BytesIO
from loguru import logger
from datetime import datetime, timedelta
from airflow.decorators import dag, task
# from airflow.providers.standard.operators.bash import BashOperator
from include.data_generator import RealisticSalesDataGenerator
from include.utils.helpers import load_config
from include.utils.s3_utils import S3Manager
from include.utils.mlflow_utils import MLFlowManager
from include.training import ModelTrainer
from include.data_loader import DataLoader

config = load_config('/usr/local/airflow/include/config.yaml')
mlflow_manager = MLFlowManager(app_config=config)
data_loader = DataLoader(app_config=config)
trainer = ModelTrainer(app_config=config)


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
    return data_loader.extract_data()

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
    return data_loader.transform_data_task(data_info=data_info)
    
  @task()
  def train_model_task(daily_store_sales: pd.DataFrame): 
    train_df, val_df, test_df = trainer.prepare_data(
      daily_store_sales, 
      target_col="sales",
      date_col="date",
      group_cols=['store_id'], 
      categorical_cols=['store_id'],
    )

    logger.info(f"Start training model with {len(train_df.columns)} features: {train_df.columns}")
    results = trainer.train(train_df, val_df, test_df, target_col="sales", use_optuna=True)

    serializable_results = {}
    for model_name, result in results.items(): 
      if "metrics" in result: 
        serializable_results[model_name] = {"metrics": result.get("metrics", {})}

    return {
      "training_results": serializable_results, 
      "mlflow_run_id": results['run_id']
    }
  
  @task
  def evaluate_models_task(training_results: dict): 
    results = training_results['training_results']

    best_rmse = float("inf")
    best_model_name = None

    for model_name, result in results.items(): 
      metric = result['metrics']
      if metric['rmse'] < best_rmse: # càng nhỏ càng tốt
        best_rmse = metric['rmse']
        best_model_name = model_name
    best_run = mlflow_manager.get_best_run(metric="rmse", ascending=True)
  
    return {
      "best_model": best_model_name, 
      "best_run_id": best_run['run_id']
    }
   
  @task
  def register_best_model_task(evaluate_results: dict): 
    model_name = evaluate_results['best_model']
    run_id = evaluate_results['best_run_id']

    results = {"model_name": model_name}
    try: 
      version = mlflow_manager.register_model(run_id=run_id, model_name=model_name)
      results['version'] = version
      logger.info(f"Registed model {model_name} version {version} sucessfully")
      return results
    except Exception as e: 
      logger.error(f"Register model failed. {str(e)}")
  
  @task
  def transition_model_task(register_result: str): 
    version = register_result['version']
    model_name = register_result['model_name']

    try: 
      mlflow_manager.transition_model_stage(model_name=model_name, version=version, stage="Production")
      logger.info(f"Transition model {model_name} v{version} to Production sucessfully")
    except Exception as e: 
      logger.error(f"Error during transition model {model_name} v{version} to Production. {str(e)}")
  
    
  # Invoke the DAG
  data_info = extract_data_task()
  validation_summary = validate_data_task(data_info=data_info)
  daily_store_sales = transform_data_task(data_info=data_info)
  train_results = train_model_task(daily_store_sales=daily_store_sales)
  evaluate_results = evaluate_models_task(training_results=train_results)
  register_results = register_best_model_task(evaluate_results=evaluate_results)
  transition_model_task(register_result=register_results)

# Instantiate the DAG
sales_forecast_training_dag = sales_forecast_training()
