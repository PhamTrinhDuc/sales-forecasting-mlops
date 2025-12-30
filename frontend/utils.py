import os
import requests
from typing import List, Dict, Any, Optional
from loguru import logger

# Base URL for the inference API
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


def health_check() -> Dict[str, Any]:
    """
    Check the health status of the inference service.
    
    Returns:
        Dict containing the health status
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
    try:
        url = f"{API_BASE_URL}/health"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Health check failed: {str(e)}")
        raise


def predict_single(
    store_id: str,
    date: str,
    additional_features: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make a single prediction for a specific store and date.
    
    Args:
        store_id: The store identifier
        date: Date in YYYY-MM-DD format
        additional_features: Optional additional features for prediction
        
    Returns:
        Dict containing prediction results with keys:
            - store_id: str
            - date: str
            - predictions: dict
            - intervals: dict
            - model_version: str
            - prediction_timestamp: str
            
    Raises:
        requests.exceptions.RequestException: If the API request fails
    """
    try:
        url = f"{API_BASE_URL}/predict/single"
        payload = {
            "store_id": store_id,
            "date": date,
        }
        if additional_features:
            payload["additional_features"] = additional_features
            
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Single prediction failed for store {store_id}: {str(e)}")
        raise


def predict_batch(requests_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Make batch predictions for multiple stores and dates.
    
    Args:
        requests_data: List of request dictionaries, each containing:
            - store_id: str
            - date: str (YYYY-MM-DD format)
            - additional_features: Optional[Dict[str, Any]]
            
    Returns:
        Dict containing batch prediction results with same structure as single prediction
        
    Raises:
        requests.exceptions.RequestException: If the API request fails
        
    Example:
        >>> requests_data = [
        ...     {"store_id": "001", "date": "2025-12-30"},
        ...     {"store_id": "002", "date": "2025-12-31"}
        ... ]
        >>> results = predict_batch(requests_data)
    """
    try:
        url = f"{API_BASE_URL}/predict/batch"
        response = requests.post(url, json=requests_data, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise


def generate_forecast_dates(start_date: str, forecast_days: int) -> List[str]:
    """
    Generate a list of dates for forecasting.
    
    Args:
        start_date: Starting date in YYYY-MM-DD format
        forecast_days: Number of days to forecast
        
    Returns:
        List of date strings in YYYY-MM-DD format
    """
    from datetime import datetime, timedelta
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [(start + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(forecast_days)]
    return dates


def check_api_health() -> bool:
    """
    Check if the API is healthy and ready.
    
    Returns:
        True if API is healthy, False otherwise
    """
    try:
        result = health_check()
        return result.get("status") == "healthy"
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        return False


