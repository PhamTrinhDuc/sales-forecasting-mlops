
# TROUBLESHOOTING

### 1 Vài lưu ý khi dùng astronomer 
1. Astro yêu cầu khởi tạo project trước sử dụng câu lệnh `astro dev init`. Nếu không có thì không thể `astro dev run`
2. Khi khởi tạo, astro sẽ tạo 1 Dockerfile và việc custom Dockerfile sẽ mở rộng image airflow của astro. Cụ thể khi chạy lightgbm, xgboot thì cần cài các thư viện C++, vì thế cần override lại Dockfile để build lại image airflow
3. File docker-compose.override.yml cũng sẽ mở rộng thêm các service chứ không làm ảnh hưởng tới các services có sẵn. Và lưu ý là không ghi đè vào các services có sẵn do astro khởi tạo 
4. Khi override thêm các serives thì cần cấu hình các services này chung network với airflow của astro. Như vậy airflow mới giao tiếp được với các services khác
5. Astro tự động nhận diện file requirements và cài các thư viện vào container airflow để đảm bảo Dag chạy được 
6. Khi cập nhật Dockerfile, thư viện trong requirements và chạy `astro dev restart` lại project thì có container có thực sự được cập nhật không? Không, vì thế ta cần: 
  - `astro dev kill` => xóa container => chạy `astro dev start` để mọi thứ khởi động lại 
  - `astro dev start -no-cache` sẽ build lại container chứ không cache
7. Khi sửa code dưới local mà airflow UI chưa load lại code thì chạy: 
  - `astro dev kill` để kill project 
  - Đổi tên file `docker-compose.override.yml` thành 1 tên khác (nếu không sẽ lỗi vì chưa có network airflow)
  - Run `astro dev run` 
  - Đổi lại tên file docker-compose như cũ rồi chạy lại `astro dev start`