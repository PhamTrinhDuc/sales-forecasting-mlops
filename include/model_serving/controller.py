from loguru import logger
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from utils import load_config
from models import RequestModel, ResponseModel
from services import ModelInferenceService

config = load_config('/usr/local/airflow/include/config.yaml')

inference_service = ModelInferenceService()

async def life_span(app: FastAPI): 
  inference_service.load_models() # load models vào RAM
  yield # Đợi và xử lý các API request.
  logger.info(f"Shutting down inference service") # Stop server (Ctrl+C): In log


app = FastAPI(
  title="Sales forecasting inference API", 
  description="API for sales forecasting model inference", 
  version="1.0.0", 
  lifespan=life_span
)

@app.get("/health")
async def health_check(): 
  return {
    "status": "healthy", 
  }

@app.post("/predict/single", response_model=ResponseModel)
async def predict_single(request: RequestModel):
  try: 
    logger.info(f"Start inference predict single")
    response = inference_service.predict_single(store_id=request.store_id, 
                                     date=request.date, 
                                     additional_features=request.additional_features)
    return response
  except Exception as e: 
    logger.error(f"Error during inference single. {str(e)}")
    raise HTTPException(status_code=404, detail="Failed inference single")

@app.post("/predict/batch", response_model=ResponseModel)
async def predict_batch(): 
  pass

