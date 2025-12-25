
import joblib
import numpy as np


class EnsembleModel: 
  def __init__(self, ensemble_models: dict, ensemble_weights: dict):
    
    self.models = ensemble_models

    if ensemble_weights is None: 
      n_models = len(ensemble_models)
      self.ensemble_weights = {model: 1 / n_models for model in ensemble_models.keys()}
    else: 
      self.ensemble_weights = ensemble_weights

    for model in ensemble_models: 
      if model not in ensemble_weights: 
        self.ensemble_weights[model] = 0.0
    
  def predict(self, X): 
    predictions = []
    weights = []

    for model_name, model in self.models.items(): 
      if self.ensemble_weights[model] > 0: 
        pred = model.predict(X)
        predictions.append(pred)
        weights.append(self.ensemble_weights[model])
    
    if not predictions: 
      raise ValueError("No models avaiable for prediction")

    predictions = np.array(predictions)
    weights = np.array(weights)

    ensem_pred = np.average(predictions, axis=0, weights=weights)
    return ensem_pred
  def get_params(self): 
    return {
      'model': self.models, 
      'weights': self.ensemble_weights
    }
  
  def set_params(self, **params): 
    for key, value in params.items(): 
      setattr(params, key, value)
    return self

  def save(self, file_path: str): 
    joblib.dump(self, file_path)
  
  @classmethod
  def load(cls, file_path: str): 
    return joblib.load(file_path)

  def __repr__(self):
    model_info = []
    for name, weight in self.ensemble_weights.items():
        model_info.append(f"{name}: {weight:.3f}")
    return f"EnsembleModel({', '.join(model_info)})"