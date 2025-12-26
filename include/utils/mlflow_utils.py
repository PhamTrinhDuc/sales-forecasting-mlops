
import os
import socket
import yaml
import joblib
import mlflow
import pandas as pd
from urllib.parse import urlparse
from mlflow.tracking import MlflowClient
from loguru import logger
from datetime import datetime
from typing import List, Optional, Dict, Any

os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "5"
os.environ["MLFLOW_HTTP_MAX_RETRIES"] = "1"
os.environ["MLFLOW_HTTP_TIMEOUT"]= "5"

class MLFlowManager: 
  def __init__(self, app_config: dict): 

    self.mlflow_config = app_config['mlflow']
    self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.mlflow_config['tracking_uri'])
    self.experiment_name = self.mlflow_config['experiment_name']
    self.registry_name = self.mlflow_config['registry_name']
    self.model_uri = self.mlflow_config['model_uri']

    logger.info(f"Connecting to MLflow at {self.tracking_uri}...")
    # Set tracking URI cho toàn bộ MLflow API
    mlflow.set_tracking_uri(self.tracking_uri)
    
    url_parsed = urlparse(url=self.tracking_uri)
    if self._is_mlflow_available(host=url_parsed.hostname, 
                                     port=url_parsed.port, 
                                     timeout=2): 
    
      self.client = MlflowClient(tracking_uri=self.tracking_uri)

      experiment = self.client.get_experiment_by_name(name=self.experiment_name)
      if experiment is None: 
        self.experiment_id = self.client.create_experiment(name=self.experiment_name)
        logger.info(f"Create experiment sucessfully: {self.experiment_name}")
      else: 
        self.experiment_id = experiment.experiment_id
        logger.info(f"Experiment already existed")
    
  def _is_mlflow_available(self, host: str, port: int, timeout: int=2): 
    """Check if Mlflow agent is reachable"""
    try: 
      sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
      socket.timeout(timeout)
      socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
      sock.close()
      return True
    
    except (socket.gaierror, socket.timeout, OSError) as e: 
      logger.warning(f"⚠️  Mlflow not available at {host}:{port} - {e}")
      return False

  def start_run(self, 
                run_name: Optional[str]=None, 
                tags: dict[str, any]=None) -> str: 
    
    if run_name is None: 
      run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run = self.client.create_run(run_name=run_name, tags=tags, experiment_id=self.experiment_id)
    run_id = run.info.run_id

    logger.info(f"Started Mlflow run with run_id: {run_id}")
    return run_id
  
  def log_param(self, params: dict, run_id: str): 
    for key, value in params.items(): 
      self.client.log_param(key=key, value=value, run_id=run_id)

  def log_metric(self, metrics: Dict[str, float], run_id: str, step: Optional[int]=None): 
    for key, value in metrics.items(): 
      self.client.log_metric(key=key, value=value, run_id=run_id, step=step)

  # def log_model(self, model_name: str, model, run_id: str):
  #   try:
  #     import mlflow.xgboost
  #     import mlflow.lightgbm
  #     import mlflow.sklearn
      
  #     # Tự động chọn flavor để log cho đúng chuẩn MLflow
  #     mlflow.set_tracking_uri(self.tracking_uri)
      
  #     with mlflow.start_run(run_id=run_id):
  #       if "xgboost" in model_name.lower():
  #         mlflow.xgboost.log_model(
  #             xgb_model=model, 
  #             artifact_path="model",  # artifact path cố định
  #             registered_model_name=f"{self.registry_name}_{model_name}" # Đăng ký luôn nếu muốn
  #         )
  #       elif "lightgbm" in model_name.lower():
  #         mlflow.lightgbm.log_model(
  #             lgb_model=model, 
  #             artifact_path="model",  # artifact path cố định
  #             registered_model_name=f"{self.registry_name}_{model_name}", 
  #         )
  #       else:
  #         # Fallback cho các model khác dùng sklearn wrapper
  #         mlflow.sklearn.log_model(
  #             sk_model=model,
  #             artifact_path="model",
  #             registered_model_name=f"{self.registry_name}_{model_name}"
  #         )
  #       # Thêm các flavor khác tương tự...
          
  #     logger.info(f"Successfully logged {model_name} as a proper MLflow Model")
  #   except Exception as e:
  #     logger.error(f"failed to log model {model_name}: {e}")
  #     raise ValueError(f"Failed log model to Mlflow. {str(e)}")

  def log_model(self, 
                model_name: str, 
                model,
                run_id: str,
                # input_examples: Optional[pd.DataFrame]=None, 
                # signature: Optional[Any]=None, 
                # registered_model_name: Optional[str]=None
                ): 
    try: 
      import tempfile
      with tempfile.TemporaryDirectory() as tempdir: 
        model_path = os.path.join(tempdir, f"{model_name}_model.pkl")
        joblib.dump(model, model_path)

        # Log artifact
        self.client.log_artifact(local_path=model_path, 
                                 run_id=run_id,
                                 artifact_path=f"models/{model_name}")
        logger.info(f"Sucessfully saved {model_name} model as artifact")

        # save metadata
        metadata = {
          "model_type": model_name, 
          "framework": type(model).__module__, 
          "class": type(model).__name__, 
          "timestamp": datetime.now().isoformat()
        }

        metadata_path = os.path.join(tempdir, f"{model_name}_metadata.yaml")
        with open(metadata_path, "w") as f: 
          yaml.dump(metadata, f)
        
        self.client.log_artifact(local_path=metadata_path, artifact_path=f"models/{model_name}", run_id=run_id)
    except Exception as e: 
      logger.error(f"failed to log model {model_name}: {e}")

  def log_artifacts(self, run_id: str, artifact_path: str): 
    self.client.log_artifacts(run_id, artifact_path)
  
  def download_artifacts_from_s3(self, run_id: str, path=".", dst_path: str="artifacts"): 
    """Default artifacs save S3 after config mlflow in docker-compose.yml"""
    os.makedirs(dst_path, exist_ok=True)
    artifacts_dir = self.client.download_artifacts(run_id=run_id, path=path, dst_path=dst_path)
    return artifacts_dir

  def get_best_model(self, metric="rmse", ascending: bool=True): 
    experiment = self.client.get_experiment_by_name(self.experiment_name)

    runs = self.client.search_runs(
      experiment_ids=[experiment.experiment_id], 
      order_by=[f"metrics.{metric} {'ASC'if ascending else 'DESC'}"], 
      max_results=1
    )
    if len(runs) == 0: 
      raise ValueError(f"No runs found in expriment: {self.experiment_name}")
    best_run = runs[0]

    return {
      "experiment_id": experiment.experiment_id,
      "run_id": best_run.info.run_id, 
      "metrics": best_run.data.metrics, 
      "params": best_run.data.params
    }
  
  def register_model(self, run_id: str, model_name: str) -> str: 
    """Register model if possible, otherwise return run_id as version"""
    try: 
      model_uri = self.model_uri.format(experiment_id=self.experiment_id, 
                                        run_id=run_id, 
                                        model_name=model_name)
      # model_uri = f"mlflow-artifacts:/{self.experiment_id}/{run_id}/artifacts/models/{model_name}"
      model = mlflow.register_model(model_uri=model_uri, 
                                    name=f"{self.registry_name}_{model_name}"
                                    )
      return model.version
    except Exception as e: 
      logger.warning(f"Error during register model, using run_id. {str(e)}")
      raise ValueError(f"Failed register model: {model_name}. {str(e)}")

  def transition_model_stage(self, model_name: str, version: str, stage: str):
    try: 
      self.client.transition_model_version_stage(
        name=f'{self.registry_name}_{model_name}',
        version=version, 
        stage=stage, 
        archive_existing_versions=True # chuyển các version cũ về archived
      )

    except Exception as e: 
      logger.warning(f"Error during transition model to {stage}. {str(e)}")

  def get_latest_model(self, model_name: str, stage: str="Production"):
    try:
      filter_string = f"name='{self.registry_name}_{model_name}'"
      versions = self.client.search_model_versions(filter_string=filter_string)
      if not versions: 
        raise ValueError(f"No model version found for {model_name}")
      
      versions = [version for version in versions if version.current_stage==stage]
      latest_version = max(versions, key=lambda x: int(x.version))

      return {
        "version": latest_version.version, 
        "stage": stage, 
        "run_id": latest_version.run_id, 
        "model_uri": latest_version.source
      }

    except Exception as e: 
      logger.warning(f"Get latest model failed. Switch to best model. {str(e)}")
      best_model = self.get_best_model()
      model_uri = self.model_uri.format(experiment_id=self.experiment_id, 
                                        run_id=best_model['run_id'], 
                                        model_name=model_name)
      return {
        "version": best_model['run_id'],
        "stage": stage, 
        "run_id": best_model['run_id'], 
        'model_uri': model_uri
      }
  
  def end_run(self, run_id: str): 
    pass


if __name__ == "__main__": 
  from dotenv import load_dotenv
  load_dotenv(".env.dev")

  from helpers import load_config
  app_config_path = "/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml"
  
  RUN_ID = "f36f6be37e9c4e9da44743a0aa40750c"
  EXPER_NAME = "sale_forecasting"

  app_config = load_config(app_config_path)
  manager = MLFlowManager(app_config=app_config)
  # dirs = manager.download_artifacts(run_id="6589e6ae9ddb4758a707cb39bd127c37")
  # print(dirs)

  # best_model = manager.get_best_model()
  # print(best_model)

  # version = manager.register_model(run_id=RUN_ID, model_name="lightgbm")
  # manager.transition_model_stage(model_name="lightgbm", version=version, stage="production")

  # result = manager.get_latest_model(model_name="xgboost", stage="Production")
  # print(result)
  # # import mlflow
  
  # version = result['version']
  # model = mlflow.xgboost.load_model(f"models:/sale_forecast_models_xgboost/{version}")

  # print(model)

  model = joblib.load("/home/ducpham/workspace/Sales-Forecasting-Mlops/include/artifacts/models/xgboost/xgboost_model.pkl")
  model_name = "xgboost"
  run_id = "064fdc854a404f6683aedbc9e10b0446"

  manager.log_model(model=model, model_name=model_name, run_id=run_id)