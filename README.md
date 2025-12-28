# Sales Forecasting MLOps Pipeline

![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)
![Airflow](https://img.shields.io/badge/Airflow-3.0.1-blue)
![MLflow](https://img.shields.io/badge/MLflow-3.0.1-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Hệ thống dự đoán doanh số bán hàng với MLOps pipeline hoàn chỉnh, sử dụng Apache Airflow để orchestration, MLflow để quản lý model lifecycle và FastAPI để phục vụ inference.

## 📋 Mục lục

- [Tổng quan](#-tổng-quan)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
- [Tính năng](#-tính-năng)
- [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Pipeline](#-pipeline)
- [Mô hình ML](#-mô-hình-ml)
- [API Documentation](#-api-documentation)
- [Monitoring](#-monitoring)
- [Troubleshooting](#-troubleshooting)

## 🎯 Tổng quan

Dự án này xây dựng một hệ thống MLOps end-to-end để dự đoán doanh số bán hàng, bao gồm:

- **Data Pipeline**: Thu thập và xử lý dữ liệu bán hàng từ nhiều nguồn (sales, promotions, customer traffic, store events, inventory)
- **Feature Engineering**: Tạo các đặc trưng thời gian, lag features và rolling statistics
- **Model Training**: Huấn luyện và tối ưu hóa các mô hình ML (XGBoost, LightGBM) với Optuna
- **Model Management**: Quản lý model lifecycle với MLflow
- **Model Serving**: Triển khai API inference với FastAPI
- **Orchestration**: Tự động hóa pipeline với Apache Airflow

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────┐      ┌──────────────┐      ┌─────────────┐
│   Data Sources  │─────▶│   MinIO/S3   │─────▶│   Airflow   │
│  (CSV/Parquet)  │      │   (Storage)  │      │   (Sched.)  │
└─────────────────┘      └──────────────┘      └──────┬──────┘
                                                       │
                         ┌─────────────────────────────┼──────────┐
                         │                             │          │
                    ┌────▼─────┐              ┌───────▼────┐ ┌───▼────┐
                    │  Extract │              │ Transform  │ │ Train  │
                    │   Data   │──────────────▶│   Data     │─▶│ Model  │
                    └──────────┘              └────────────┘ └────┬───┘
                                                                   │
                    ┌──────────┐              ┌────────────┐      │
                    │ Register │◀─────────────│  Evaluate  │◀─────┘
                    │  Model   │              │   Models   │
                    └────┬─────┘              └────────────┘
                         │
                    ┌────▼─────┐              ┌────────────┐
                    │  MLflow  │──────────────▶│  FastAPI   │
                    │ Registry │              │ Inference  │
                    └──────────┘              └────────────┘
```

## ✨ Tính năng

### Data Management
- ✅ Tự động tạo dữ liệu mô phỏng với `RealisticSalesDataGenerator`
- ✅ Lưu trữ dữ liệu trên MinIO (S3-compatible storage)
- ✅ Validation dữ liệu tự động (kiểm tra missing columns, negative values)
- ✅ Xử lý nhiều loại dữ liệu: sales, promotions, customer traffic, store events, inventory

### Feature Engineering
- ✅ **Date Features**: year, month, day, dayofweek, quarter, weekofyear, is_weekend, is_holiday (Vietnam holidays)
- ✅ **Lag Features**: 1, 2, 3, 7, 14, 21, 30 days lag
- ✅ **Rolling Features**: mean, std, min, max, median với windows 3, 7, 14, 21, 30 days
- ✅ Aggregation từ product-level lên store-level

### Model Training
- ✅ Hỗ trợ nhiều mô hình: XGBoost, LightGBM, Ensemble
- ✅ Hyperparameter tuning với Optuna (30 trials mặc định)
- ✅ Cross-validation với 5 folds
- ✅ Early stopping để tránh overfitting
- ✅ Đánh giá với nhiều metrics: RMSE, MAE, MAPE, R²

### MLOps
- ✅ Experiment tracking với MLflow
- ✅ Model versioning và registry
- ✅ Artifact storage trên S3
- ✅ Automated pipeline với Airflow (schedule @weekly)
- ✅ Model comparison và selection

### Model Serving
- ✅ FastAPI REST API
- ✅ Single và batch prediction
- ✅ Health check endpoint
- ✅ Async inference

## 🛠️ Công nghệ sử dụng

| Component | Technology | Version |
|-----------|-----------|---------|
| **Orchestration** | Apache Airflow | 3.0.1 |
| **ML Tracking** | MLflow | 3.0.1 |
| **Storage** | MinIO (S3-compatible) | latest |
| **Database** | PostgreSQL | 12.6 |
| **ML Models** | XGBoost, LightGBM | - |
| **Optimization** | Optuna | 4.6.0+ |
| **API Framework** | FastAPI | 0.117.1+ |
| **Data Processing** | Pandas, NumPy | - |
| **Language** | Python | 3.12+ |

## 🚀 Cài đặt

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Astro CLI (cho Airflow development)

### 1. Clone repository

```bash
git clone <repository-url>
cd Sales-Forecasting-Mlops
```

### 2. Cài đặt dependencies

```bash
# Sử dụng uv package manager
uv sync

# Hoặc sử dụng pip
pip install -e .
```

### 3. Khởi động các services với Docker Compose

```bash
# Khởi động Airflow và các services phụ trợ
astro dev start

# Services sẽ được khởi động:
# - Airflow Webserver: http://localhost:8080
# - MLflow UI: http://localhost:5001
# - MinIO Console: http://localhost:9001
# - PostgreSQL: localhost:5432
```

### 4. Cấu hình

Tạo file `.env.dev` trong thư mục `include/`:

```env
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
AWS_DEFAULT_REGION=us-east-1
MLFLOW_TRACKING_URI=http://mlflow:5001
```

Chỉnh sửa `include/config.yaml` theo nhu cầu:

```yaml
dataset:
  data_bucket: 'data-sales-forecasting'
  start_date: '2025-01-01'
  end_date: '2025-12-31'

training:
  optuna_trials: 30
  train_size: 0.7
  val_size: 0.15
```

## 📖 Sử dụng

### 1. Chạy Training Pipeline

#### Qua Airflow UI

1. Truy cập http://localhost:8080
2. Đăng nhập (username/password: admin/admin)
3. Bật DAG `sales_forecast_training`
4. Trigger DAG manually hoặc chờ schedule (@weekly)

#### Qua CLI

```bash
# Trigger DAG
astro dev run dags trigger sales_forecast_training

# Xem logs
astro dev logs scheduler
```

### 2. Theo dõi Training với MLflow

```bash
# Truy cập MLflow UI
open http://localhost:5001
```

Tại đây bạn có thể:
- Xem các experiments và runs
- So sánh metrics giữa các models
- Download artifacts (models, scalers, encoders)
- Xem hyperparameters

### 3. Deploy Model Inference API

```bash
cd include/model_serving

# Chạy FastAPI server
uvicorn controller:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Sử dụng API

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "store_001",
    "date": "2025-12-28",
    "features": {
      "month": 12,
      "dayofweek": 6,
      "is_weekend": 1,
      "sales_lag_7": 1500.0
    }
  }'
```

#### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "store_id": "store_001",
      "date": "2025-12-28",
      "features": {...}
    },
    {
      "store_id": "store_002",
      "date": "2025-12-28",
      "features": {...}
    }
  ]'
```

## 📁 Cấu trúc dự án

```
Sales-Forecasting-Mlops/
│
├── dags/                          # Airflow DAGs
│   └── sales_forecast_training.py # Main training pipeline DAG
│
├── include/                       # Core modules
│   ├── config.yaml               # Configuration file
│   ├── data_generator.py         # Synthetic data generation
│   ├── data_loader.py            # Data extraction & transformation
│   ├── feature_pipeline.py       # Feature engineering
│   ├── training.py               # Model training logic
│   │
│   ├── ml_models/                # Model implementations
│   │   ├── ensemble_model.py
│   │   ├── comparation_model.py
│   │   └── visualization_model.py
│   │
│   ├── evaluate/                 # Model evaluation
│   │   └── diagnostic.py
│   │
│   ├── utils/                    # Utilities
│   │   ├── helpers.py
│   │   ├── mlflow_utils.py
│   │   └── s3_utils.py
│   │
│   ├── model_serving/            # API serving
│   │   ├── controller.py         # FastAPI endpoints
│   │   ├── services.py           # Inference logic
│   │   └── models.py             # Pydantic models
│   │
│   └── artifacts/                # Saved models & preprocessors
│       ├── models/
│       └── preprocessor/
│
├── data/                         # Data storage
│   ├── sales/
│   ├── promotions/
│   ├── customer_traffic/
│   ├── store_events/
│   └── inventory/
│
├── docs/                         # Documentation
│   ├── training.md
│   └── troubleshooting.md
│
├── tests/                        # Unit tests
│
├── docker-compose.override.yml   # Additional Docker services
├── Dockerfile                    # Custom Airflow image
├── pyproject.toml               # Python dependencies
└── README.md                    # This file
```

## 🔄 Pipeline

### Sales Forecast Training DAG

Pipeline chạy theo lịch **@weekly** và bao gồm các bước:

```
1. Extract Data
   ├── Load từ MinIO/S3
   └── Generate synthetic data nếu cần

2. Validate Data
   ├── Check required columns
   ├── Check data types
   ├── Check value ranges
   └── Log validation issues

3. Transform Data
   ├── Merge sales với promotions
   ├── Aggregate từ product-level → store-level
   └── Create daily_store_sales dataframe

4. Train Models
   ├── Feature Engineering (date, lag, rolling)
   ├── Split train/val/test (70/15/15)
   ├── Hyperparameter tuning với Optuna
   ├── Train XGBoost
   ├── Train LightGBM
   ├── Train Ensemble model
   └── Log to MLflow

5. Evaluate Models
   ├── Compare RMSE, MAE, MAPE, R²
   ├── Select best model
   └── Get best run from MLflow

6. Register Model
   └── Register best model to MLflow Model Registry
```

### Task Dependencies

```python
data_info = extract_data_task()
validate_summary = validate_data_task(data_info)
daily_store_sales = transform_data_task(data_info)
training_results = train_model_task(daily_store_sales)
evaluate_results = evaluate_models_task(training_results)
register_best_model_task(evaluate_results)
```

## 🤖 Mô hình ML

### XGBoost

```yaml
params:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  objective: "reg:squarederror"
  random_state: 42
```

**Optuna tuning**:
- `max_depth`: [3, 10]
- `learning_rate`: [0.01, 0.3]
- `n_estimators`: [50, 300]
- `subsample`: [0.6, 1.0]
- `colsample_bytree`: [0.6, 1.0]

### LightGBM

```yaml
params:
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  objective: "regression"
  random_state: 42
```

**Optuna tuning**:
- `num_leaves`: [20, 100]
- `learning_rate`: [0.01, 0.3]
- `n_estimators`: [50, 300]
- `min_child_samples`: [5, 50]

### Ensemble Model

Kết hợp XGBoost và LightGBM với weighted average:
- Weights được tối ưu hóa dựa trên validation performance
- Improved robustness và accuracy

### Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **RMSE** | Root Mean Squared Error | Minimize |
| **MAE** | Mean Absolute Error | Minimize |
| **MAPE** | Mean Absolute Percentage Error | Minimize |
| **R²** | R-squared Score | Maximize |

## 📊 API Documentation

### Endpoints

#### `GET /health`

Health check endpoint

**Response:**
```json
{
  "status": "healthy"
}
```

#### `POST /predict/single`

Dự đoán doanh số cho một store/ngày

**Request Body:**
```json
{
  "store_id": "store_001",
  "date": "2025-12-28",
  "features": {
    "month": 12,
    "dayofweek": 6,
    "is_weekend": 1,
    "sales_lag_7": 1500.0,
    "sales_rolling_mean_7": 1450.0
  }
}
```

**Response:**
```json
{
  "prediction": 1520.5,
  "model_version": "v1.0.0",
  "timestamp": "2025-12-28T10:30:00"
}
```

#### `POST /predict/batch`

Dự đoán batch cho nhiều stores/ngày

**Request Body:**
```json
[
  {
    "store_id": "store_001",
    "date": "2025-12-28",
    "features": {...}
  },
  {
    "store_id": "store_002",
    "date": "2025-12-28",
    "features": {...}
  }
]
```

## 📈 Monitoring

### MLflow Tracking

```bash
# Truy cập MLflow UI
http://localhost:5001

# Xem experiments
Experiments → sales_forecasting

# Compare runs
Select multiple runs → Compare
```

### Airflow Monitoring

```bash
# Airflow UI
http://localhost:8080

# Xem DAG runs
DAGs → sales_forecast_training → Graph/Calendar

# Xem logs
Click vào task → Logs
```

### MinIO Storage

```bash
# MinIO Console
http://localhost:9001

# Login
Username: minioadmin
Password: minioadmin

# Buckets
- data-sales-forecasting: Raw data
- mlflow-artifacts: Models & artifacts
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Airflow DAG không hiển thị

```bash
# Check scheduler logs
astro dev logs scheduler

# Restart scheduler
astro dev restart scheduler
```

#### 2. MLflow connection error

```bash
# Kiểm tra MLflow service
docker ps | grep mlflow

# Check environment variables
echo $MLFLOW_TRACKING_URI
```

#### 3. MinIO bucket không tồn tại

```bash
# Tạo bucket manually
docker exec -it <minio-container> mc mb myminio/data-sales-forecasting
```

#### 4. Model inference lỗi

```bash
# Check model files
ls -la include/artifacts/models/

# Verify model loading
python -c "import mlflow; print(mlflow.sklearn.load_model('path/to/model'))"
```

### Logs Location

```bash
# Airflow logs
astro dev logs <service_name>

# Application logs
include/logs/

# MLflow logs
docker logs <mlflow-container>
```

## 📝 Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_feature_pipeline.py

# With coverage
pytest --cov=include tests/
```

### Adding New Features

1. **Thêm date feature mới**: Chỉnh sửa `include/feature_pipeline.py`
2. **Thêm model mới**: Tạo class trong `include/ml_models/`
3. **Thêm data source**: Cập nhật `data_loader.py` và `config.yaml`

### Code Style

```bash
# Format code
black include/ dags/

# Lint
flake8 include/ dags/

# Type checking
mypy include/
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Jiyuu** - *Initial work* - duc78240@gmail.com

## 🙏 Acknowledgments

- Apache Airflow team
- MLflow team
- XGBoost và LightGBM contributors
- Optuna team

---

**📧 Contact**: duc78240@gmail.com  
**🔗 Project Link**: [https://github.com/yourusername/Sales-Forecasting-Mlops](https://github.com/yourusername/Sales-Forecasting-Mlops)
