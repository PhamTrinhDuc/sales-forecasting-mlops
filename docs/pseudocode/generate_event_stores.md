
```bash
FUNCTION generate_store_events() RETURNS DataFrame:
    // Khởi tạo danh sách rỗng để lưu các sự kiện cửa hàng
    INITIALIZE empty list events
    
    // ===== PHẦN 1: TẠO SỰ KIỆN CHO TỪNG CỬA HÀNG =====
    
    FOR EACH (store_id, store_info) IN stores:
        
        // --- PHẦN 1A: TẠO NGÀY ĐÓNG CỬA NGẪU NHIÊN ---
        // Mỗi cửa hàng có 2-5 ngày đóng cửa (do thời tiết, sự cố kỹ thuật, v.v.)
        SET n_closures = random_integer(2, 5)
        
        // Tạo danh sách n_closures ngày phân bố đều trong khoảng thời gian
        SET closure_dates = create_evenly_distributed_dates(
            start_date, 
            end_date, 
            n_closures
        )
        
        // Thêm mỗi ngày đóng cửa vào danh sách sự kiện
        FOR EACH date IN closure_dates:
            ADD to events:
                {
                    'store_id': store_id,
                    'date': date,
                    'event_type': 'closure',
                    'impact': -1.0        // -100% = đóng cửa hoàn toàn
                }
        
        
        // --- PHẦN 1B: TẠO SỰ KIỆN SỬA CHỮA/CẢI TẠO ---
        // Mỗi cửa hàng có 30% xác suất được sửa chữa/cải tạo
        SET random_value = random_float(0.0, 1.0)
        
        IF random_value < 0.3:    // 30% cơ hội
            // Chọn ngày bắt đầu sửa chữa ngẫu nhiên
            // Trong khoảng 100-600 ngày sau start_date
            SET random_days = random_integer(100, 600)
            SET renovation_start = start_date + random_days days
            
            // Thời gian sửa chữa từ 7-21 ngày
            SET renovation_duration = random_integer(7, 21)
            
            // Tạo sự kiện cho mỗi ngày sửa chữa
            FOR d FROM 0 TO renovation_duration-1:
                SET reno_date = renovation_start + d days
                
                // Chỉ thêm nếu ngày nằm trong khoảng thời gian hợp lệ
                IF reno_date <= end_date:
                    ADD to events:
                        {
                            'store_id': store_id,
                            'date': reno_date,
                            'event_type': 'renovation',
                            'impact': -0.3    // -30% doanh thu do sửa chữa
                        }
    
    // ===== PHẦN 2: TRẢ VỀ KẾT QUẢ =====
    
    // Chuyển list thành DataFrame và trả về
    RETURN convert_to_dataframe(events)

END FUNCTION
```