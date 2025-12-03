```bash
FUNCTION generate_sales_data(output_dir = "/tmp/sales_data") RETURNS Dict:
    
    // ===== PHẦN 0: KHỞI TẠO =====
    
    CREATE directory output_dir if not exists
    
    // Tạo dữ liệu phụ trợ
    SET promotions_df = generate_promotions()
    SET store_events_df = generate_store_events()
    
    // Khởi tạo dict để theo dõi file paths
    INITIALIZE file_paths = {
        'sales': [],
        'inventory': [],
        'customer_traffic': [],
        'promotions': [],
        'store_events': []
    }
    
    // Lưu dữ liệu phụ trợ
    SAVE promotions_df to "promotions/promotions.parquet"
    ADD path to file_paths['promotions']
    
    SAVE store_events_df to "store_events/events.parquet"
    ADD path to file_paths['store_events']
    
    
    // ===== PHẦN 1: VÒNG LẶP CHÍNH - TỪNG NGÀY =====
    
    SET current_date = start_date
    
    WHILE current_date <= end_date:
        SET date_str = format_date(current_date, 'YYYY-MM-DD')
        LOG "Generating data for {date_str}"
        
        // Khởi tạo danh sách dữ liệu cho ngày hiện tại
        INITIALIZE daily_sales_data = []
        INITIALIZE daily_traffic_data = []
        INITIALIZE daily_inventory_data = []
        
        
        // ===== PHẦN 2: VÒNG LẶP CỬA HÀNG =====
        
        FOR EACH (store_id, store_info) IN stores:
            
            // --- 2A: TÍNH CÁC YẾU TỐ ẢNH HƯỞNG ĐỀN TRAFFIC CỬA HÀNG ---
            
            // Yếu tố 1: Traffic cơ bản của cửa hàng
            SET base_traffic = store_info['base_traffic']
            // Ví dụ: New York (large) = 1000, Phoenix (small) = 400
            
            
            // Yếu tố 2: Ngày trong tuần (Day of Week Factor)
            SET dow_factor = get_day_of_week_factor(current_date)
            /*
             * LOGIC BUSINESS: 
             * - Thứ 2-4 (Mon-Wed): 0.85-0.9 (ít người)
             * - Thứ 5 (Thu): 0.9 (tăng nhẹ)
             * - Thứ 6 (Fri): 1.1 (tăng mạnh)
             * - Thứ 7 (Sat): 1.3 (cao nhất - cuối tuần)
             * - Chủ nhật (Sun): 1.2 (cao)
             */
            
            
            // Yếu tố 3: Ngày lễ (Holiday Factor)
            SET is_holiday = check_if_us_holiday(current_date)
            IF is_holiday:
                SET holiday_factor = 1.3  // Tăng 30% traffic
            ELSE:
                SET holiday_factor = 1.0
            
            
            // Yếu tố 4: Thời tiết (Weather Factor)
            SET weather_factor = random_normal(mean=1.0, stddev=0. 1)
            /*
             * LOGIC BUSINESS:
             * - Thời tiết tốt: 1.1-1.2
             * - Thời tiết bình thường: 0.9-1.1
             * - Thời tiết xấu: 0. 5-0.8
             * - Clamp trong khoảng [0.5, 1.2]
             */
            SET weather_factor = CLAMP(weather_factor, 0.5, 1.2)
            
            
            // Yếu tố 5: Sự kiện cửa hàng (Store Events)
            SET store_event_impact = 1.0  // Mặc định không ảnh hưởng
            
            IF store_events_df is not empty:
                FIND event WHERE:
                    event.store_id == store_id AND
                    event.date == current_date
                
                IF event exists:
                    SET store_event_impact = 1.0 + event.impact
                    /*
                     * LOGIC BUSINESS:
                     * - Closure: 1.0 + (-1.0) = 0. 0 → Đóng cửa hoàn toàn
                     * - Renovation: 1.0 + (-0.3) = 0.7 → Giảm 30%
                     */
            
            
            // --- 2B: TÍNH TỔNG TRAFFIC CỬA HÀNG ---
            
            SET store_traffic = ROUND(
                base_traffic * 
                dow_factor * 
                holiday_factor * 
                weather_factor * 
                store_event_impact * 
                random_normal(1.0, 0.05)  // Thêm noise nhỏ ±5%
            )
            /*
             * VÍ DỤ TÍNH TOÁN:
             * Store: New York (base=1000)
             * Date: Saturday, Black Friday, Good weather, No events
             * 
             * store_traffic = 1000 * 1.3 (Sat) * 1.3 (Holiday) * 1.1 (Weather) 
             *                 * 1.0 (No event) * 1.02 (noise)
             *               = 1,897 customers
             */
            
            
            // --- 2C: LƯU DỮ LIỆU TRAFFIC ---
            
            ADD to daily_traffic_data:
                {
                    'date': current_date,
                    'store_id': store_id,
                    'customer_traffic': store_traffic,
                    'weather_impact': weather_factor,
                    'is_holiday': is_holiday
                }
            
            
            // ===== PHẦN 3: VÒNG LẶP SẢN PHẨM =====
            
            FOR EACH (product_id, product_info) IN all_products:
                
                // --- 3A: TÍNH YẾU TỐ MÙA VỤ (SEASONALITY) ---
                
                SET seasonality_factor = get_seasonality_factor(
                    current_date, 
                    product_info['seasonality']
                )
                /*
                 * LOGIC BUSINESS - SEASONALITY:
                 * 
                 * 1.  HOLIDAY (Electronics, Home):
                 *    - Nov-Dec: 1.5-1.8 (peak shopping season)
                 *    - Major holidays: 1.3
                 *    - Other times: 1.0
                 * 
                 * 2. SUMMER (Clothing - T-shirt, Sports - Bicycle):
                 *    - June-August: 1.2-1.4
                 *    - Using sine wave: 1.0 + 0.4*sin(2π*(day-80)/365)
                 * 
                 * 3.  WINTER (Clothing - Jacket):
                 *    - Dec-Feb: 1.2-1.3
                 *    - Using sine wave: 1.0 + 0.3*sin(2π*(day+90)/365)
                 * 
                 * 4.  BACK_TO_SCHOOL (Laptops):
                 *    - August-September: 1.4
                 *    - Other times: 0.9
                 * 
                 * 5. FITNESS (Yoga Mat, Dumbbells):
                 *    - January (New Year resolution): 1.5
                 *    - May-June (summer prep): 1.3
                 *    - Other times: 1.0
                 * 
                 * 6. SPRING (Vacuum, Running Shoes):
                 *    - March-May: 1. 2-1.3
                 * 
                 * 7.  ALL_YEAR (Jeans, Air Purifier):
                 *    - Always: 1.0
                 */
                
                
                // --- 3B: KIỂM TRA KHUYẾN MÃI (PROMOTION) ---
                
                SET promotion_factor = 1.0
                SET discount_percent = 0.0
                
                IF promotions_df is not empty:
                    FIND promo WHERE:
                        promo.date == current_date AND
                        promo.product_id == product_id
                    
                    IF promo exists:
                        SET discount_percent = promo.discount_percent
                        SET promotion_factor = 1.0 + (discount_percent * 3)
                        /*
                         * LOGIC BUSINESS - PROMOTION IMPACT:
                         * 
                         * Khi giảm giá → Tăng nhu cầu (demand elasticity)
                         * Hệ số nhân: 3x
                         * 
                         * Ví dụ:
                         * - Giảm 10% (0.1) → demand tăng 30% (1.3x)
                         * - Giảm 20% (0. 2) → demand tăng 60% (1.6x)
                         * - Giảm 25% (0.25) → demand tăng 75% (1.75x)
                         * 
                         * Giải thích: Khách hàng nhạy cảm với giá, giảm giá
                         * khuyến khích mua nhiều hơn bình thường
                         */
                
                
                // --- 3C: TÍNH YẾU TỐ CỬA HÀNG & GIÁ (STORE & PRICE) ---
                
                // Yếu tố kích thước cửa hàng
                SET size_factor = MAP store_info['size']:
                    'large'  → 1.0
                    'medium' → 0.7
                    'small'  → 0.5
                /*
                 * LOGIC BUSINESS:
                 * Cửa hàng lớn có nhiều traffic và không gian hơn
                 * → Bán được nhiều sản phẩm hơn
                 */
                
                // Yếu tố giá (Price sensitivity)
                SET price_factor = 1.0 / (1.0 + product_info['price'] / 100)
                /*
                 * LOGIC BUSINESS - PRICE ELASTICITY:
                 * 
                 * Sản phẩm đắt → Ít người mua hơn (tỷ lệ chuyển đổi thấp)
                 * 
                 * Ví dụ:
                 * - T-Shirt ($29): 1.0/(1+0.29) = 0.775 (77.5% conversion)
                 * - Laptop ($999): 1.0/(1+9.99) = 0.091 (9.1% conversion)
                 * - Bicycle ($399): 1.0/(1+3.99) = 0.200 (20% conversion)
                 * 
                 * Formula giải thích:
                 * - price/100 normalize giá
                 * - Càng đắt, mẫu số càng lớn → price_factor càng nhỏ
                 */
                
                
                // --- 3D: TÍNH SỐ LƯỢNG BÁN (QUANTITY SOLD) ---
                
                // Tỷ lệ chuyển đổi cơ bản: 0.1% traffic
                SET base_quantity = store_traffic * 0.001 * size_factor * price_factor
                /*
                 * LOGIC BUSINESS:
                 * 
                 * Base conversion rate: 0.1% (1 trong 1000 khách mua sản phẩm này)
                 * 
                 * Ví dụ:
                 * - Store traffic: 1000 khách
                 * - T-Shirt ($29) tại Large store:
                 *   = 1000 * 0.001 * 1.0 * 0.775
                 *   = 0. 775 sản phẩm base
                 * 
                 * - Laptop ($999) tại Small store:
                 *   = 400 * 0.001 * 0.5 * 0.091
                 *   = 0.018 sản phẩm base
                 */
                
                // Áp dụng tất cả các yếu tố
                SET quantity = ROUND(
                    base_quantity * 
                    seasonality_factor * 
                    promotion_factor *
                    random_normal(1.0, 0.2)  // Thêm random noise ±20%
                )
                /*
                 * VÍ DỤ ĐẦY ĐỦ:
                 * 
                 * Sản phẩm: Smart Watch ($299)
                 * Store: New York (large, traffic=1500)
                 * Date: Black Friday, có promotion 25% off
                 * Seasonality: Holiday (1.5x)
                 * 
                 * Tính toán:
                 * - price_factor = 1/(1+2.99) = 0.251
                 * - base_quantity = 1500 * 0.001 * 1.0 * 0.251 = 0.377
                 * - promotion_factor = 1 + (0.25 * 3) = 1.75
                 * - seasonality_factor = 1.5
                 * 
                 * quantity = 0.377 * 1.5 * 1.75 * 1.05 (noise)
                 *          = 1.04 ≈ 1 sản phẩm
                 * 
                 * So sánh ngày thường (không promotion, no seasonality):
                 * quantity = 0.377 * 1.0 * 1.0 * 1.0 = 0.377 ≈ 0 sản phẩm
                 * 
                 * → Black Friday + Promotion tăng sales từ 0 lên 1! 
                 */
                
                SET quantity = MAX(0, quantity)  // Đảm bảo không âm
                
                
                // --- 3E: TÍNH TOÁN TÀI CHÍNH (FINANCIAL CALCULATIONS) ---
                
                IF quantity > 0:
                    // Giá thực tế sau giảm
                    SET actual_price = product_info['price'] * (1 - discount_percent)
                    /*
                     * Ví dụ: $299 * (1 - 0.25) = $224.25
                     */
                    
                    // Doanh thu
                    SET revenue = quantity * actual_price
                    /*
                     * Revenue = số lượng bán * giá bán
                     * Ví dụ: 1 * $224.25 = $224.25
                     */
                    
                    // Chi phí (Cost of Goods Sold - COGS)
                    SET cost = quantity * product_info['price'] * (1 - product_info['margin'])
                    /*
                     * LOGIC BUSINESS - MARGIN:
                     * 
                     * margin = tỷ lệ lợi nhuận trên giá gốc
                     * cost = giá gốc * (1 - margin)
                     * 
                     * Ví dụ Smart Watch:
                     * - Giá bán: $299
                     * - Margin: 20% (0.2)
                     * - Cost per unit: $299 * (1-0.2) = $239.2
                     * - Profit per unit: $299 - $239.2 = $59.8
                     * 
                     * Khi có promotion 25% off:
                     * - Giá bán: $224.25
                     * - Cost: vẫn $239.2 (chi phí không đổi)
                     * - Profit: $224.25 - $239.2 = -$14.95 (LỖ!)
                     * 
                     * → Strategy: Chấp nhận lỗ để tăng volume và market share
                     */
                    
                    // Lợi nhuận
                    SET profit = revenue - cost
                    
                    // Lưu dữ liệu bán hàng
                    ADD to daily_sales_data:
                        {
                            'date': current_date,
                            'store_id': store_id,
                            'product_id': product_id,
                            'category': product_info['category'],
                            'quantity_sold': quantity,
                            'unit_price': product_info['price'],
                            'discount_percent': discount_percent,
                            'revenue': revenue,
                            'cost': cost,
                            'profit': profit
                        }
                
                
                // --- 3F: QUẢN LÝ TỒN KHO (INVENTORY MANAGEMENT) ---
                
                // Mức tồn kho hiện tại
                SET inventory_level = random_integer(50, 200)
                /*
                 * LOGIC BUSINESS:
                 * Random 50-200 đơn vị (simplified)
                 * 
                 * Trong thực tế nên:
                 * - Track theo ngày trước
                 * - inventory_today = inventory_yesterday - quantity_sold + restock
                 */
                
                // Điểm đặt hàng lại (Reorder Point)
                SET reorder_point = random_integer(20, 50)
                /*
                 * LOGIC BUSINESS:
                 * 
                 * Khi inventory_level <= reorder_point → Đặt hàng thêm
                 * 
                 * Reorder point phụ thuộc:
                 * - Lead time (thời gian nhận hàng)
                 * - Average daily sales
                 * - Safety stock
                 * 
                 * Formula thực tế:
                 * ROP = (Average Daily Sales * Lead Time) + Safety Stock
                 */
                
                // Số ngày tồn kho còn lại
                SET days_of_supply = inventory_level / MAX(1, quantity)
                /*
                 * LOGIC BUSINESS:
                 * 
                 * Days of Supply = Số ngày có thể bán với tốc độ hiện tại
                 * 
                 * Ví dụ:
                 * - Inventory: 100 units
                 * - Daily sales: 5 units
                 * - Days of supply: 100/5 = 20 days
                 * 
                 * Dùng để:
                 * - Dự báo khi nào hết hàng
                 * - Quyết định đặt hàng
                 */
                
                ADD to daily_inventory_data:
                    {
                        'date': current_date,
                        'store_id': store_id,
                        'product_id': product_id,
                        'inventory_level': inventory_level,
                        'reorder_point': reorder_point,
                        'days_of_supply': days_of_supply
                    }
            
            // END FOR EACH product
        
        // END FOR EACH store
        
        
        // ===== PHẦN 4: LƯU DỮ LIỆU THEO NGÀY =====
        
        // --- 4A: LƯU SALES DATA (MỖI NGÀY) ---
        IF daily_sales_data is not empty:
            CONVERT daily_sales_data to DataFrame sales_df
            
            // Tạo đường dẫn phân vùng theo năm/tháng/ngày
            SET sales_path = output_dir + 
                "/sales/year={year}/month={month:02d}/day={day:02d}/" +
                "sales_{date_str}.parquet"
            
            CREATE directories if not exist
            SAVE sales_df to sales_path
            ADD sales_path to file_paths['sales']
        
        
        // --- 4B: LƯU CUSTOMER TRAFFIC (MỖI NGÀY) ---
        IF daily_traffic_data is not empty:
            CONVERT daily_traffic_data to DataFrame traffic_df
            
            SET traffic_path = output_dir + 
                "/customer_traffic/year={year}/month={month:02d}/day={day:02d}/" +
                "traffic_{date_str}. parquet"
            
            CREATE directories if not exist
            SAVE traffic_df to traffic_path
            ADD traffic_path to file_paths['customer_traffic']
        
        
        // --- 4C: LƯU INVENTORY (MỖI CHỦ NHẬT) ---
        SET day_of_week = get_day_of_week(current_date)
        
        IF daily_inventory_data is not empty AND day_of_week == SUNDAY:
            /*
             * LOGIC BUSINESS:
             * 
             * Inventory snapshot chỉ lưu mỗi tuần (Chủ nhật)
             * - Giảm storage cost
             * - Vẫn đủ để phân tích xu hướng
             * - Weekly reporting cycle
             */
            
            CONVERT daily_inventory_data to DataFrame inventory_df
            
            GET iso_week = get_iso_week_number(current_date)
            
            SET inventory_path = output_dir + 
                "/inventory/year={year}/week={week:02d}/" +
                "inventory_{date_str}.parquet"
            
            CREATE directories if not exist
            SAVE inventory_df to inventory_path
            ADD inventory_path to file_paths['inventory']
        
        
        // Chuyển sang ngày tiếp theo
        SET current_date = current_date + 1 day
    
    // END WHILE (main date loop)
    
    
    // ===== PHẦN 5: TẠO METADATA =====
    
    SET metadata = {
        'generation_date': current_timestamp(),
        'start_date': start_date,
        'end_date': end_date,
        'n_stores': count(stores),
        'n_products': count(all_products),
        'file_counts': {
            'sales': count(file_paths['sales']),
            'inventory': count(file_paths['inventory']),
            'customer_traffic': count(file_paths['customer_traffic']),
            'promotions': count(file_paths['promotions']),
            'store_events': count(file_paths['store_events'])
        },
        'total_files': sum of all file counts
    }
    
    SAVE metadata to "metadata/generation_metadata.parquet"
    
    LOG "Generated {total_files} files"
    LOG "Sales files: {sales_count}"
    LOG "Output directory: {output_dir}"
    
    
    // ===== PHẦN 6: TRẢ VỀ KẾT QUẢ =====
    
    RETURN file_paths

END FUNCTION
```