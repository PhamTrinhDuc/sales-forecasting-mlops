
```bash
FUNCTION generate_promotions() RETURNS DataFrame:
    INITIALIZE empty list promotions
    
    DEFINE major_events = [
        ('Black Friday', 11, 4, 5, 0.25),
        ('Cyber Monday', 11, 4, 2, 0.20),
        ('Christmas Sale', 12, 15, 10, 0.15),
        ('New Year Sale', 1, 1, 7, 0. 20),
        ('Presidents Day', 2, 15, 3, 0.15),
        ('Memorial Day', 5, 25, 3, 0. 15),
        ('July 4th Sale', 7, 1, 5, 0.15),
        ('Labor Day', 9, 1, 3, 0.15),
        ('Back to School', 8, 1, 14, 0. 10)
    ]
    
    // ✅ FIX: Lặp qua từng năm trong khoảng thời gian
    GET start_year = extract_year(start_date)
    GET end_year = extract_year(end_date)
    
    FOR year FROM start_year TO end_year:
        FOR EACH (event_name, month, day, duration, discount) IN major_events:
            
            IF event_name == 'Black Friday':
                SET november = first_day_of_november(year)
                GET all_thursdays = find_all_thursdays_in_month(november)
                SET event_date = all_thursdays[3] + 1 day
            ELSE:
                TRY:
                    SET event_date = create_date(year, month, day)
                CATCH exception:
                    CONTINUE to next event
            
            // Kiểm tra event_date có nằm trong khoảng [start_date, end_date]
            IF start_date <= event_date <= end_date:
                FOR d FROM 0 TO duration-1:
                    SET promo_date = event_date + d days
                    
                    IF promo_date <= end_date:
                        SET num_products = random_integer(5, 15)
                        SET promo_products = random_sample(all_products, num_products)
                        
                        FOR EACH product_id IN promo_products:
                            ADD to promotions:
                                {
                                    'date': promo_date,
                                    'product_id': product_id,
                                    'promotion_type': event_name,
                                    'discount_percent': discount
                                }
    
    // Flash sales (giữ nguyên)
    SET total_days = (end_date - start_date).days
    SET n_flash_sales = total_days * 0.05
    SET flash_dates = create_evenly_distributed_dates(start_date, end_date, n_flash_sales)
    
    FOR EACH date IN flash_dates:
        SET num_products = random_integer(3, 8)
        SET promo_products = random_sample(all_products, num_products)
        
        FOR EACH product_id IN promo_products:
            SET random_discount = random_float(0.1, 0.3)
            
            ADD to promotions:
                {
                    'date': date,
                    'product_id': product_id,
                    'promotion_type': 'Flash Sale',
                    'discount_percent': random_discount
                }
    
    RETURN convert_to_dataframe(promotions)

END FUNCTION
```