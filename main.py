from include.utils.helpers import load_config
from include.utils.s3_utils import S3Manager
import pandas as pd
from io import BytesIO
from loguru import logger

config = load_config('/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml')
bucket_name = config['dataset']['data_bucket']

s3_manager = S3Manager(config_app=config, 
                       endpoint_url="http://localhost:9000", 
                       aws_access_key_id="minioadmin", 
                       aws_secret_access_key="minioadmin")

data_info = s3_manager.get_files(bucket_name=bucket_name)


def transform_data_task(data_info: dict) -> pd.DataFrame: 
    # 1. 
    file_paths = data_info
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


if __name__ == "__main__": 
  csv = transform_data_task(data_info=data_info)
  
  csv['sale_lag_7'] = csv.groupby("store_id")['sales'].shift(7)
  print(csv['sale_lag_7'])