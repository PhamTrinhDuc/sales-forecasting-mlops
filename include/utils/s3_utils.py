import boto3
from botocore.config import Config
import os
from loguru import logger


class S3Manager: 
    def __init__(self, 
                 config_app,
                 endpoint_url: str = None, 
                 aws_access_key_id: str = None, 
                 aws_secret_access_key: str = None):
      self.endpoint_url = endpoint_url  
      self.config_data = config_app['dataset']

      try:
        self.s3_client = boto3.client(
          's3',
          endpoint_url=endpoint_url or os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000'),
          aws_access_key_id=aws_access_key_id or os.getenv('AWS_ACCESS_KEY_ID', 'minioadmin'),
          aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY', 'minioadmin'),
          config=Config(signature_version='s3v4', connect_timeout=5, retries={'max_attempts': 1}),
          region_name="us-east-1",
        )
        logger.info("S3 client initialized successfully.")
      
      except self.s3_client.exceptions.TimeoutError as e:
        logger.error(f"Timeout error initializing S3 client: {e}")
        raise e
      except Exception as e:
        logger.error(f"Error initializing S3 client: {e}")
        raise e
    
    def create_bucket(self, bucket_name: str): 
      try:
        self.s3_client.create_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} created successfully.")
      except self.s3_client.exceptions.BucketAlreadyOwnedByYou:
        logger.info(f"Bucket {bucket_name} already exists and is owned by you.")
      except Exception as e:
        logger.error(f"Error creating bucket {bucket_name}: {e}")
        raise e


    def delete_bucket(self, bucket_name: str): 
      try:
        self.s3_client.delete_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} deleted successfully.")
      except Exception as e:
        logger.error(f"Error deleting bucket {bucket_name}: {e}")
        raise e


    def validate_object_in_bucket(self, bucket_name: str, data_types: list[str]=None) -> bool: 
      response = self.s3_client.list_objects(Bucket=bucket_name)
      list_objects = response.get('Contents', [])
      list_object_keys = [obj['Key'] for obj in list_objects]
      object_types = set( file_path.split("/")[0] for file_path in list_object_keys)

      data_types = data_types or self.config_data['data_types']

      if all(data_type in data_types for data_type in data_types) and len(list_object_keys) > 0: 
        logger.info(f"All expected data types found in bucket {bucket_name}.")
        return True
      else:
        raise ValueError(
          f"Not all data types found in bucket {bucket_name}. Expected: {data_types}, Found: {object_types}"
        )


    def upload_file(self, file_path: str, bucket_name: str, s3_key: str):
      try:
        self.s3_client.upload_file(file_path, bucket_name, s3_key)
      except Exception as e:
        logger.error(f"Failed to upload {file_path} to {bucket_name}/{s3_key}: {e}")
        raise e


    def upload_folders(self, folder_path: str, bucket_name: str): 

      logger.info(f"Uploading files from folder: {folder_path}")
      count_sccess = 0
      count_fail = 0
      for root, _, files in os.walk(folder_path):
        for file in files:
          file_path = os.path.join(root, file)
          s3_key = os.path.relpath(file_path, folder_path)
          try:
            self.upload_file(file_path=file_path, bucket_name=bucket_name, s3_key=s3_key)
            count_sccess += 1
          except Exception as e:
            count_fail += 1

      logger.info(f"Upload completed: {count_sccess} files succeeded, {count_fail} files failed.")


    def upload_objects(self, file_bytes: bytes, bucket_name: str, s3_key: str):
      try:
        self.s3_client.put_object(Body=file_bytes, Bucket=bucket_name, Key=s3_key)
      except Exception as e:
        logger.error(f"Failed to upload object to {bucket_name}/{s3_key}: {e}")
        raise e
      
      
    def get_files(self, bucket_name: str, include_url: bool = True): 
      """Get files from bucket with direct URLs (no expiration)"""
      files = {
        "customer_traffic": [], 
        "inventory": [], 
        "promotions": [], 
        "sales": [], 
        "store_events": []
      }

      try:
        response = self.s3_client.list_objects(Bucket=bucket_name)
        # Get endpoint URL
        endpoint = self.s3_client.meta.endpoint_url or self.endpoint_url
        
        for obj in response.get('Contents', []):
          key = obj['Key']
          file_data = {"key": key}
          
          # Generate direct URL (s3://bucket/key format or http format)
          if include_url:
            file_data["url"] = f"{endpoint}/{bucket_name}/{key}"
            file_data["s3_path"] = f"s3://{bucket_name}/{key}"
          
          # Categorize by folder
          if key.endswith("parquet"): # đảm bảo các file là file parquet 
            if "customer_traffic" in key:
              files["customer_traffic"].append(file_data)
            elif "inventory" in key:
              files["inventory"].append(file_data)
            elif "promotions" in key:
              files["promotions"].append(file_data)
            elif "sales" in key:
              files["sales"].append(file_data)
            elif "store_events" in key:
              files["store_events"].append(file_data)
          else: 
            continue
            
        logger.info(f"Files retrieved from bucket {bucket_name} successfully.")
        return files
      
      except Exception as e:
        logger.error(f"Error retrieving files from bucket {bucket_name}: {e}")
        raise e
      

    def read_file_as_bytes(self, bucket_name: str, s3_key: str):
      """Read S3 file as bytes (works with boto3 credentials)"""
      try:
        response = self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)
        data = response['Body'].read()
        return data
      except Exception as e:
        logger.error(f"Failed to read {s3_key}: {e}")
        raise e


if __name__ == "__main__":

  config_path = '/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml'
  from helpers import load_config
  config_app = load_config(config_path=config_path)
  BUCKET_NAME = config_app['dataset']['data_bucket']

  s3_manager = S3Manager(
    config_app=config_app,
    endpoint_url="http://localhost:9000",
    aws_access_key_id="minioadmin",
    aws_secret_access_key="minioadmin",
  )
  # s3_manager.delete_bucket(bucket_name=BUCKET_NAME)
  # s3_manager.create_bucket(bucket_name=BUCKET_NAME, public_read=True)
  # s3_manager.upload_folders(folder_path="/home/ducpham/workspace/Sales-Forecasting-Mlops/data", 
  #                           bucket_name=BUCKET_NAME)

  # Get files with URLs
  # files = s3_manager.get_files(bucket_name=BUCKET_NAME, include_url=True)
  # s3_key = files['sales'][0]['key']
  # file_bytes = s3_manager.read_file_as_bytes(bucket_name=BUCKET_NAME, s3_key=s3_key)
  
  # import pandas as pd
  # from io import BytesIO
  # df = pd.read_csv(BytesIO(file_bytes))
  # print(df.head())
  
  # Get just keys without URLs
  # files_no_url = s3_manager.get_files(bucket_name=BUCKET_NAME, include_url=False)
  # print(f"\nTotal sales files: {len(files_no_url['sales'])}")

  is_bucket_exists = s3_manager.validate_object_in_bucket(bucket_name=BUCKET_NAME)
