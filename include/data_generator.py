import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List
import holidays
from loguru import logger
from include.utils.s3_utils import S3Manager



class RealisticSalesDataGenerator:
    """Generate realistic sales data with multiple files, partitions, and business patterns"""
    
    def __init__(self, config_app: Dict):

        self.config_app = config_app['dataset']
        self.start_date = self.config_app['start_date']
        self.end_date = self.config_app['end_date']

        self.start_date = pd.to_datetime(self.start_date)
        self.end_date = pd.to_datetime(self.end_date)
        self.vn_holidays = holidays.VN()

        self.s3_manager = S3Manager(config_app=config_app)
        
        # Store configurations
        self.stores : dict[str, dict] = {
          "store_001": {"location": "Hà Nội", "size": "large", "base_traffic": 1200},
          "store_002": {"location": "Hồ Chí Minh", "size": "large", "base_traffic": 1100},
          "store_003": {"location": "Đà Nẵng", "size": "medium", "base_traffic": 800},
          "store_004": {"location": "Hải Phòng", "size": "medium", "base_traffic": 900},
          "store_005": {"location": "Cần Thơ", "size": "small", "base_traffic": 500},
          "store_006": {"location": "Nha Trang", "size": "medium", "base_traffic": 700},
          "store_007": {"location": "Vũng Tàu", "size": "small", "base_traffic": 400},
          "store_008": {"location": "Buôn Ma Thuột", "size": "small", "base_traffic": 300},
          "store_009": {"location": "Đà Lạt", "size": "medium", "base_traffic": 600},
          "store_010": {"location": "Quy Nhơn", "size": "medium", "base_traffic": 650},
        }
        
        # Product categories and items
        self.product_categories = {
            'Electronics': {
                'ELEC_001': {'name': 'Smartphone', 'price': 699, 'margin': 0.15, 'seasonality': 'holiday'},
                'ELEC_002': {'name': 'Laptop', 'price': 999, 'margin': 0.12, 'seasonality': 'back_to_school'},
                'ELEC_003': {'name': 'Headphones', 'price': 199, 'margin': 0.25, 'seasonality': 'holiday'},
                'ELEC_004': {'name': 'Tablet', 'price': 499, 'margin': 0.18, 'seasonality': 'holiday'},
                'ELEC_005': {'name': 'Smart Watch', 'price': 299, 'margin': 0.20, 'seasonality': 'fitness'}
            },
            'Clothing': {
                'CLTH_001': {'name': 'T-Shirt', 'price': 29, 'margin': 0.50, 'seasonality': 'summer'},
                'CLTH_002': {'name': 'Jeans', 'price': 79, 'margin': 0.45, 'seasonality': 'all_year'},
                'CLTH_003': {'name': 'Jacket', 'price': 149, 'margin': 0.40, 'seasonality': 'winter'},
                'CLTH_004': {'name': 'Dress', 'price': 89, 'margin': 0.48, 'seasonality': 'summer'},
                'CLTH_005': {'name': 'Shoes', 'price': 119, 'margin': 0.42, 'seasonality': 'all_year'}
            },
            'Home': {
                'HOME_001': {'name': 'Coffee Maker', 'price': 79, 'margin': 0.30, 'seasonality': 'holiday'},
                'HOME_002': {'name': 'Blender', 'price': 49, 'margin': 0.35, 'seasonality': 'summer'},
                'HOME_003': {'name': 'Vacuum Cleaner', 'price': 199, 'margin': 0.28, 'seasonality': 'spring'},
                'HOME_004': {'name': 'Air Purifier', 'price': 149, 'margin': 0.32, 'seasonality': 'all_year'},
                'HOME_005': {'name': 'Toaster', 'price': 39, 'margin': 0.40, 'seasonality': 'holiday'}
            },
            'Sports': {
                'SPRT_001': {'name': 'Yoga Mat', 'price': 29, 'margin': 0.55, 'seasonality': 'fitness'},
                'SPRT_002': {'name': 'Dumbbells', 'price': 49, 'margin': 0.45, 'seasonality': 'fitness'},
                'SPRT_003': {'name': 'Running Shoes', 'price': 129, 'margin': 0.38, 'seasonality': 'spring'},
                'SPRT_004': {'name': 'Bicycle', 'price': 399, 'margin': 0.25, 'seasonality': 'summer'},
                'SPRT_005': {'name': 'Tennis Racket', 'price': 89, 'margin': 0.35, 'seasonality': 'summer'}
            }
        }
        
        # Flatten products
        self.all_products = {}
        for category, products in self.product_categories.items():
            for product_id, product_info in products.items():
                self.all_products[product_id] = {**product_info, 'category': category}
    
    def get_seasonality_factor(self, date: pd.Timestamp, seasonality_type: str) -> float:
        """Calculate seasonality factor based on date and type"""
        day_of_year = date.dayofyear
        
        if seasonality_type == 'holiday':
            # Peak during November-December and around major holidays
            if date.month in [11, 12]:
                return 1.5 + 0.3 * np.sin(2 * np.pi * (day_of_year - 300) / 60)
            elif date in self.vn_holidays:
                return 1.3
            else:
                return 1.0
        
        elif seasonality_type == 'summer':
            # Peak June-August
            return 1.0 + 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        elif seasonality_type == 'winter':
            # Peak December-February
            return 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year + 90) / 365)
        
        elif seasonality_type == 'back_to_school':
            # Peak August-September
            if date.month in [8, 9]:
                return 1.4
            else:
                return 0.9
        
        elif seasonality_type == 'fitness':
            # Peak January (New Year) and May-June (summer prep)
            if date.month == 1:
                return 1.5
            elif date.month in [5, 6]:
                return 1.3
            else:
                return 1.0
        
        elif seasonality_type == 'spring':
            # Peak March-May
            return 1.0 + 0.3 * np.sin(2 * np.pi * (day_of_year - 20) / 365)
        
        else:  # all_year
            return 1.0
    
    def get_day_of_week_factor(self, date: pd.Timestamp) -> float:
        """Get multiplier based on day of week"""
        dow = date.dayofweek
        # Monday=0, Sunday=6
        dow_factors = [0.9, 0.85, 0.85, 0.9, 1.1, 1.3, 1.2]
        return dow_factors[dow]
    
    def generate_promotions(self) -> pd.DataFrame:
        """
        REFERENCE TỚI MÃ GIẢ TẠI: docs/pseudocode/generate_promotions.md

        1. Tạo các sự kiện khuyến mãi lớn trong năm
        - Tạo trước 1 danh sách các sự kiện lớn trong năm (fixed)
        - Xét trong khoảng thời gian từ ngày bắt đầu đến ngày kết thúc: 
          + Lấy ra ngày sự kiện (nếu là black friday thì lấy thứ 6 cuối của tháng, còn không thì tạo dựa vào ngày tháng của danh sách)
          + Đảm bảo sự kiện diễn ra trong khoảng thời gian này
          + Duyệt qua thời gian diễn ra của sự kiện (cập nhật lại ngày sự kiện đang diễn ra)
            + Random 1 vài sản phẩm để khuyến mãi 
            + Tạo bản ghi khuyến mãi cho từng ngày
            + Output là dict dạng [date, product_id, promotion_type, discount_percent]
        2. Tạo các sự kiện khuyến mãi ngẫu nhiên nhỏ lẻ
          + tạo 1 list danh sách các ngày flashsale ngẫu nhiên (4% tổng ngày start -> end)
          + Duyệt qua các ngày: 
            + Random danh sách sản phẩm (3-8 sp)
            + Duyệt qua danh sách sản phẩm vào tạo promotion cho sản phẩm
        """
        promotions = []
        
        # Major sales events
        major_events = [
            ('Black Friday', 11, 4, 5, 0.25),  # 4th Friday of November, 5 days, 25% off
            ('Cyber Monday', 11, 4, 2, 0.20),  # Monday after Black Friday
            ('Christmas Sale', 12, 15, 10, 0.15),
            ('New Year Sale', 1, 1, 7, 0.20),
            ('Presidents Day', 2, 15, 3, 0.15),
            ('Memorial Day', 5, 25, 3, 0.15),
            ('July 4th Sale', 7, 1, 5, 0.15),
            ('Labor Day', 9, 1, 3, 0.15),
            ('Back to School', 8, 1, 14, 0.10),
        ]
        
        current_date = self.start_date
        while current_date <= self.end_date:
            year = current_date.year
            
            for event_name, month, day, duration, discount in major_events:
                if event_name == 'Black Friday':
                    # Calculate 4th Thursday of November, then add 1 for Friday
                    november = pd.Timestamp(year, 11, 1)
                    thursdays = pd.date_range(november, november + timedelta(days=30), freq='W-THU')
                    event_date = thursdays[3] + timedelta(days=1)
                else:
                    try:
                        event_date = pd.Timestamp(year, month, day)
                    except:
                        continue
                
                if self.start_date <= event_date <= self.end_date:
                    for d in range(duration):
                        promo_date = event_date + timedelta(days=d)
                        if promo_date <= self.end_date:
                            # Random products on promotion
                            promo_products = random.sample(list(self.all_products.keys()), 
                                                         k=random.randint(5, 15))
                            for product_id in promo_products:
                                promotions.append({
                                    'date': promo_date,
                                    'product_id': product_id,
                                    'promotion_type': event_name,
                                    'discount_percent': discount
                                })
            
            current_date = current_date + pd.DateOffset(years=1)
        
        # Add random flash sales
        n_flash_sales = int((self.end_date - self.start_date).days * 0.05)  # 5% of days
        flash_dates = pd.date_range(self.start_date, self.end_date, periods=n_flash_sales)
        
        for date in flash_dates:
            promo_products = random.sample(list(self.all_products.keys()), k=random.randint(3, 8))
            for product_id in promo_products:
                promotions.append({
                    'date': date,
                    'product_id': product_id,
                    'promotion_type': 'Flash Sale',
                    'discount_percent': random.uniform(0.1, 0.3)
                })
        
        return pd.DataFrame(promotions)
    
    def generate_store_events(self) -> pd.DataFrame:
        """
        REFERENCE TỚI MÃ GIẢ TẠI: docs/pseudocode/generate_store_events.md

        - Khởi tạo list lưu sự kiện cho các cửa hàng
        - Output: [store_id, date, event_type, impact]
        1. Tạo sự kiện đóng cửa: đóng cửa trong 1 ngày
        - Duyệt qua mỗi cửa hàng: 
        + random 2-5 ngày đóng cửa ngẫu nhiên. Tạo danh sách các ngày này 
        + Duyệt qua các ngày đóng cửa => thêm bản ghi vào events
        2. Tạo các sự kiện sửa chữa: random < 0.3 thì tạo skien này
        - Tạo ngày bắt đầu (từ 100-600 sau ngày start). Tạo danh sách ngày sửa (7-21)
        - Duyệt qua từng ngày và thêm sự kiện cho từng cửa hàng
        """
        events = []
        
        for store_id, store_info in self.stores.items():
            # Random store closures (weather, technical issues)
            n_closures = random.randint(2, 3)
            closure_dates = pd.date_range(self.start_date, self.end_date, periods=n_closures)
            
            for date in closure_dates:
                events.append({
                    'store_id': store_id,
                    'date': date,
                    'event_type': 'closure',
                    'impact': -1.0  # 100% reduction
                })
            
            # Store renovations (longer impact)
            if random.random() < 0.3:  # 30% chance of renovation
                renovation_start = self.start_date + timedelta(days=random.randint(100, 600))
                renovation_duration = random.randint(5, 7)
                
                for d in range(renovation_duration):
                    reno_date = renovation_start + timedelta(days=d)
                    if reno_date <= self.end_date:
                        events.append({
                            'store_id': store_id,
                            'date': reno_date,
                            'event_type': 'renovation',
                            'impact': -0.3  # 30% reduction
                        })
        
        return pd.DataFrame(events)
    
    def generate_sales_data(self, output_dir: str) -> Dict[str, List[str]]:
        """
        REFERENCE TỚI MÃ GIẢ TẠI: docs/pseudocode/generate_sales_data.md

        - Khởi tạo dict lưu trữ path files
        1. Khởi tạo 
          - Khởi tạo 2 datafram: promotions và store_events lưu lại dưới dạng parquet
        2. Duyệt qua từng ngày để tạo dữ liệu (sales, inventory, traffic)
          - Chuyển ngày qua dạng (YYYY-MM-DD)
          - Khởi tạo danh sách lưu dữ liệu cho ngày hiện tại (sales, traffic, inventory)
          2.1 Duyệt qua từng cửa hàng 
            - Lấy ra các loại traffic bao gồm: 
              + traffic có sẵn của cửa hàng 
              + traffic ngày trong tuần (day_of_weeek_factor)
              + traffic ngày lễ
              + traffic thời tiết 
              + traffic sự kiện của cửa hàng (closure, renovation)
            - Tính tổng traffic theo công thức:
                store_traffic = ROUND(traffic có sẵn *
                                      hệ só ngày trong tuần *
                                      hệ số ngày lễ *
                                      hệ số thời tiết *
                                      hệ số tác động tới của hàng *
                                      random.normal(1.0, 0.05) # thêm noise nhỏ ±20%
                                )
            - Lưu dữ liệu traffic vào list: {date, store_id, customer_traffic, weather_impact, is_holiday}
          2.2 Duyệt qua từn sản phẩm 
            - 2.2.1. Tính yêu tố mùa vụ 
              + Tính hệ số mùa vụ (get_seasionlity_factor)
            - 2.2.2. Kiểm tra khuyến mãi
              + Tìm promotions dựa vào: date và product_id để => discount_percent và promotion_factor
              + Map size cửa hàng sang hệ số (large -> 1.0, medium -> 0.7, small -> 0.5)
              + Tính hệ số giá ( 1/ (1 + price/100)) => cao khi sản phẩm giá rẻ
            - 2.2.3. Tính số lượng bán
              + Tính số lượng bán: base_quantity= traffic_store*0.001*size_store*price_factor. 
                  quantity = ROUND(
                                  giá cơ bản * 
                                  hệ số thời vụ *
                                  hệ số giảm giá * 
                                  random noise ±20%
                                  )
                  quantity = max(quantity, 0) # đảm bảo kh âm
            - 2.2.4. Tính tài chính
              + Tính giá thực tế sau khi giảm discount
              + Tính doanh thu = số lượng * price 
              + Chi phí: cost = giá gốc / (1- margin) # margin là con số đã được ước lượng trước dựa trên nhiều yếu tố (mặt bằng, nhân viên, v.v..) từ đó suy ra chi phí bán của sản phẩm để mang lại lợi nhuận
              + profit = cost - giá gốc
            - 2.2.5. Lưu dữ liệu bán hàng: (date, store_id, category, quantity_sold, unit_price, discount_percent, revenue, profit)
            - 2.2.6. Tính toán tồn kho 
              + random số lượng tông kho (20-50)
              + Tạo mốc số lượng reorder lại hàng 
              + Số lượng ngày tiêu thụ hàng: số lượng tồn kho / số lượng bán
              + Thêm dữ liệu (date, store_id, product_id, inventory, reorder_point, days_of_supply)

        3. Lưu dữ liệu dạng parquet theo ngày
          - 3.1 Lưu dữ liệu sales: "sales/year={year}/month={month}/day={day}/sales_{date_str}.parquet"
          - 3.2 Lưu dữ liệu customer traffic: "customer_traffic/year={year}/month={month}/day={day}/traffic+{date_str}.parquet"
          - 3.3 Lưu dữ liệu inventory (mỗi CN): "inventory/year={year}/week={week}/inventory_{date_str}.parquet"
        """

        os.makedirs(output_dir, exist_ok=True)
        
        # Generate supplementary data
        promotions_df = self.generate_promotions()
        store_events_df = self.generate_store_events()
        
        # Track file paths
        file_paths = {
            'sales': [],
            'inventory': [],
            'customer_traffic': [],
            'promotions': [],
            'store_events': []
        }
        
        # Save supplementary data
        promotions_path = os.path.join(output_dir, "promotions/promotions.parquet")
        os.makedirs(os.path.dirname(promotions_path), exist_ok=True)
        promotions_df.to_parquet(promotions_path, index=False)
        file_paths['promotions'].append(promotions_path)
        
        events_path = os.path.join(output_dir, "store_events/events.parquet")
        os.makedirs(os.path.dirname(events_path), exist_ok=True)
        store_events_df.to_parquet(events_path, index=False)
        file_paths['store_events'].append(events_path)
        
        # Generate sales data by day (more realistic for production)
        current_date = self.start_date
        
        while current_date <= self.end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            logger.info(f"Generating data for {date_str}")
            
            # Daily sales data for all stores
            daily_sales_data = []
            daily_traffic_data = []
            daily_inventory_data = []
            issued_quantity = {}
            
            # Generate data for each store for this specific day
            for store_id, store_info in sorted(self.stores.items(), reverse=True):
                # Store-level factors
                base_traffic = store_info['base_traffic']
                
                # Date factors
                dow_factor = self.get_day_of_week_factor(current_date)
                is_holiday = current_date in self.vn_holidays
                holiday_factor = 1.3 if is_holiday else 1.0
                
                # Weather impact (random)
                weather_factor = np.random.normal(1.0, 0.1)
                weather_factor = max(0.5, min(1.2, weather_factor))
                
                # Check for store events
                store_event_impact = 1.0
                if not store_events_df.empty:
                    event = store_events_df[
                        (store_events_df['store_id'] == store_id) & 
                        (store_events_df['date'] == current_date)
                    ]
                    if not event.empty:
                        store_event_impact = 2.0 + event.iloc[0]['impact'] # đảm bảo > 0 (impact là số âm)
                
                # Calculate store traffic
                store_traffic = int(
                    base_traffic * dow_factor * holiday_factor * 
                    weather_factor * store_event_impact * np.random.normal(1.2, 0.05)
                )

                # logger.debug(f" Base traffic {base_traffic}, DOW factor {dow_factor}, Holiday factor {holiday_factor}, Weather factor {weather_factor}, Event impact {store_event_impact} => Traffic {store_traffic}")
                
                daily_traffic_data.append({
                    'date': current_date,
                    'store_id': store_id,
                    'customer_traffic': store_traffic,
                    'weather_impact': weather_factor,
                    'is_holiday': is_holiday
                })

                issued_quantity = {date_str: []}
                
                # Generate product-level sales
                for product_id, product_info in self.all_products.items():
                    # Product seasonality
                    seasonality_factor = self.get_seasonality_factor(
                        current_date, product_info['seasonality']
                    )
                    
                    # Check for promotions
                    promotion_factor = 1.0
                    discount_percent = 0.0
                    if not promotions_df.empty:
                        promo = promotions_df[
                            (promotions_df['date'] == current_date) & 
                            (promotions_df['product_id'] == product_id)
                        ]
                        if not promo.empty:
                            discount_percent = promo.iloc[0]['discount_percent']
                            # Promotion increases demand
                            promotion_factor = 1.0 + (discount_percent * 3)  # 3x multiplier
                    
                    # Calculate sales quantity
                    # Base conversion rate depends on store size and product price
                    size_factor = {'large': 1.0, 'medium': 0.7, 'small': 0.5}[store_info['size']]
                    price_factor = 0.5 + 0.5 * np.exp(-product_info['price'] / 500)
                    base_quantity = store_traffic * 0.003 * size_factor * price_factor

                    # Debug
                    # logger.debug(f"store traffic: {store_traffic} Size factor: {size_factor}, Price factor: {price_factor}, Base quantity: {base_quantity}")
                    
                    quantity = int(
                        base_quantity * seasonality_factor * promotion_factor *
                        np.random.normal(1.2, 0.05)
                    )

                    # Debug
                    # logger.debug(f"Base quantity {base_quantity}, Seasonality {seasonality_factor}, Promotion {promotion_factor}, Final quantity {quantity}")

                    # Calculate revenue
                    actual_price = product_info['price'] * (1 - discount_percent)
                    revenue = quantity * actual_price
                    cost = quantity * product_info['price'] * (1 - product_info['margin'])
                    
                    if quantity > 0:
                        daily_sales_data.append({
                            'date': current_date,
                            'store_id': store_id,
                            'product_id': product_id,
                            'category': product_info['category'],
                            'quantity_sold': quantity,
                            'unit_price': product_info['price'],
                            'discount_percent': discount_percent,
                            'revenue': revenue,
                            'cost': cost,
                            'profit': revenue - cost
                        })
                    # Debug
                    # else: 
                    #     issued_quantity[date_str].append(f"Zero quantity: {quantity} for product {product_id} at store {store_id} on {date_str}")
                    
                    # Inventory tracking
                    inventory_level = random.randint(50, 200)
                    reorder_point = random.randint(20, 50)
                    
                    daily_inventory_data.append({
                        'date': current_date,
                        'store_id': store_id,
                        'product_id': product_id,
                        'inventory_level': inventory_level,
                        'reorder_point': reorder_point,
                        'days_of_supply': inventory_level / max(1, quantity)
                    })

            # Debug
            # if issued_quantity[date_str]:
            #     for issue in issued_quantity[date_str]:
            #         logger.warning(f"- {issue}")
            # break


            # Save daily files with proper partitioning
            # Sales data - one file per day
            if daily_sales_data:
                sales_df = pd.DataFrame(daily_sales_data)
                sales_path = os.path.join(
                    output_dir, 
                    f"sales/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/"
                    f"sales_{date_str}.parquet"
                )
                os.makedirs(os.path.dirname(sales_path), exist_ok=True)
                sales_df.to_parquet(sales_path, index=False)
                file_paths['sales'].append(sales_path)
            
            # Customer traffic data - one file per day
            if daily_traffic_data:
                traffic_df = pd.DataFrame(daily_traffic_data)
                traffic_path = os.path.join(
                    output_dir,
                    f"customer_traffic/year={current_date.year}/month={current_date.month:02d}/day={current_date.day:02d}/"
                    f"traffic_{date_str}.parquet"
                )
                os.makedirs(os.path.dirname(traffic_path), exist_ok=True)
                traffic_df.to_parquet(traffic_path, index=False)
                file_paths['customer_traffic'].append(traffic_path)
            
            # Inventory data - daily snapshots
            if daily_inventory_data and current_date.dayofweek == 6:  # Weekly on Sundays
                inventory_df = pd.DataFrame(daily_inventory_data)
                inventory_path = os.path.join(
                    output_dir,
                    f"inventory/year={current_date.year}/week={current_date.isocalendar()[1]:02d}/"
                    f"inventory_{date_str}.parquet"
                )
                os.makedirs(os.path.dirname(inventory_path), exist_ok=True)
                inventory_df.to_parquet(inventory_path, index=False)
                file_paths['inventory'].append(inventory_path)
            
            # Move to next day
            current_date = current_date + timedelta(days=1)
        
        # Generate metadata
        metadata = {
            'generation_date': datetime.now().isoformat(),
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'n_stores': len(self.stores),
            'n_products': len(self.all_products),
            'file_counts': {k: len(v) for k, v in file_paths.items()},
            'total_files': sum(len(v) for v in file_paths.values())
        }
        
        metadata_df = pd.DataFrame([metadata])
        metadata_path = os.path.join(output_dir, "metadata/generation_metadata.parquet")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        metadata_df.to_parquet(metadata_path, index=False)

        # upload to s3
        self.s3_manager.upload_folders(folder_path=output_dir, bucket_name=self.config_app['data_bucket'])
        logger.info(f"Generated {metadata['total_files']} files")
        logger.info(f"Sales files: {len(file_paths['sales'])}")
        logger.info(f"Output directory: {output_dir}")
        
        return file_paths

    
if __name__ == "__main__": 
  FOLDER_DIR = "/home/ducpham/workspace/Sales-Forecasting-Mlops/data"

  from utils.helpers import load_config
  config_app = load_config("/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml")

  generator = RealisticSalesDataGenerator(config_app=config_app)
  # promotions = generator.generate_promotions(store_path="./promotions.csv")
  # store_events = generator.generate_store_events(store_path="./data/csv/store_events.csv")
  generator.generate_sales_data(output_dir=FOLDER_DIR)

  inventory_df = pd.read_parquet(path=FOLDER_DIR + "/inventory/year=2025/week=01/inventory_2025-01-05.parquet")
  sales_df = pd.read_parquet(path=FOLDER_DIR + "/sales/year=2025/month=01/day=02/sales_2025-01-02.parquet")
  traffic_df = pd.read_parquet(path=FOLDER_DIR + "/customer_traffic/year=2025/month=01/day=01/traffic_2025-01-01.parquet")
  metata_df = pd.read_parquet(path=FOLDER_DIR + "/metadata/generation_metadata.parquet")

  inventory_df.to_csv(FOLDER_DIR + "/csv/inventory.csv")
  sales_df.to_csv(FOLDER_DIR + "/csv/sales.csv")
  traffic_df.to_csv(FOLDER_DIR + "/csv/traffic.csv")
  metata_df.to_csv(FOLDER_DIR + "/csv/metadata.csv")