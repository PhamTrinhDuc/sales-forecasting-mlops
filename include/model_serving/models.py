
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, validator
from datetime import datetime


class ResponseModel(BaseModel): 
  store_id: str
  date: str
  predictions: dict 
  intervals: dict
  model_version: str 
  prediction_timestamp: str


class RequestModel(BaseModel): 
  store_id: str 
  # product_id: str 
  date: str
  additional_features: Optional[Dict[str, Any]] = None

  @validator("date")
  def validate_data(cls, v): 
    try: 
      datetime.strptime(v, "%Y-%m-%d")
      return v
    except Exception as e: 
      raise ValueError("Date must be in YYYY-MM-DD format")