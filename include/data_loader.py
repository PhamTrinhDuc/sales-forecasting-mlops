import pandas as pd
from io import BytesIO
from loguru import logger
from data_generator import RealisticSalesDataGenerator
from utils import S3Manager


class DataLoader: 
  def __init__(self, app_config: dict):
    self.app_config = app_config
    self.data_config = app_config['dataset']

    self.s3_manager = S3Manager(config_app=app_config)

  def extract_data(self): 
    output_data_path = self.data_config['data_path']
    bucket_name = self.data_config['data_bucket']

    generator = RealisticSalesDataGenerator(config_app=self.app_config) 
       
    if self.s3_manager.validate_object_in_bucket(bucket_name=bucket_name):
      data_path_files = self.s3_manager.get_files(bucket_name=bucket_name, include_url=True)
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
  
  def transform_data_task(self, data_info: dict) -> pd.DataFrame: 
    """
    Chuẩn bị dữ liệu cho model. Biến đổi từ dữ liệu thô (product-level) sang dữ liệu tổng hợp (store-level).
    1. Group by và agg dữ liệu sales cho từng sản phẩm 
    2. Merge với promotions, tạo cột has_promotion = 1. Khi merge với sales, sản phẩm không có trong promotions thì có has_promotion = 0
    3. Group by và agg dữ liệu tổng hợp cho từng cửa hàng. Sau đo merge với dữ liệu sales
    4. Chuyển dữ liệu product-level sang dữ liệu store-level
    """

    bucket_name = self.data_config['data_bucket']
    # 1. 
    file_paths = data_info['file_paths']
    sale_dfs = []
    for sale_path in file_paths['sales']: 
      sale_df = self.s3_manager.read_file_as_bytes(
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
      promotion_df = self.s3_manager.read_file_as_bytes(
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
        traffic_df = self.s3_manager.read_file_as_bytes(
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

    # 5. Sau quá trình training và nhận diagnostic thì loại bỏ 3 feature: profit, customer_traffic, quantity_sold vì bị data leakage
    daily_store_sales = daily_store_sales.drop(columns=['profit', 'quantity_sold', 'customer_traffic'])

    return daily_store_sales
