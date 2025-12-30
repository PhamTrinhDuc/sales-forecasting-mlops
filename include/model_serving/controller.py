import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
from loguru import logger
from fastapi import FastAPI, HTTPException, BackgroundTasks
from contextlib import asynccontextmanager
from typing import List
from dotenv import load_dotenv
from utils import load_config
from models import RequestModel, ResponseModel
from services import ModelInferenceService

# config = load_config('/usr/local/airflow/include/config.yaml')
config = load_config('/home/ducpham/workspace/Sales-Forecasting-Mlops/include/config.yaml')
load_dotenv("../.env.dev")

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
    response = await inference_service.async_predict_single(request=request)
    return response
  except Exception as e: 
    logger.error(f"Error during inference single. {str(e)}")
    raise HTTPException(status_code=404, detail="Failed inference single")

@app.post("/predict/batch", response_model=List[ResponseModel])
async def predict_batch(requests: List[RequestModel]): 
  try: 
    logger.info(f"Start inference predict batch")
    response = await inference_service.async_predict_batch(requests=requests)
    return response
  except Exception as e: 
    logger.error(f"Error during inference batch. {str(e)}")
    raise HTTPException(status_code=404, detail="Failed inference batch")


# uvicorn controller:app --host 0.0.0.0 --port 8000 --reload 