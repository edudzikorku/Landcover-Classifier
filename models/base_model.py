from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.model_selection import RandomizedSearchCV, KFold # type: ignore

class BaseModel(ABC):
    """Base class for ML models - focused on core model operations only"""
    
    def __init__(self, random_state: int = 42, n_splits: int = 10):
        self.model = None
        self.best_params = None
        self.scaler = StandardScaler()
        self.random_state = random_state
        self.n_splits = n_splits
        self.class_weight = None
        self.kfold = KFold(n_splits = n_splits, shuffle = True, random_state = random_state)

    @abstractmethod
    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Define parameter grid for hyperparameter tuning"""
        pass

    def set_class_weight(self, class_weight: Optional[Dict[int, float]]) -> None:
        """Set class weight for imbalanced data"""
        self.class_weight = class_weight # type: ignore

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Preprocess the input data by scaling"""
        return self.scaler.fit_transform(X)

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> None:
        """Tune hyperparameters using RandomizedSearchCV"""
        param_grid = self.get_param_grid()
        
        # If class weight is available, include it in the parameter grid search 
        if self.class_weight is not None and 'class_weight' in param_grid:
            param_grid['class_weight'] = [None, 'balanced', self.class_weight]
        elif self.class_weight is not None and 'class_weight' not in param_grid:
            param_grid['class_weight'] = [None, 'balanced', self.class_weight]

        grid_search = RandomizedSearchCV(
            estimator = self.model,
            param_distributions = param_grid,
            cv = self.kfold,
            scoring = "accuracy",
            n_jobs = 1,
            verbose = 1,
            n_iter = 10
        )
        
        grid_search.fit(X, y)
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model"""
        self.model.fit(X, y) # type: ignore

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        # Include class weight if available and the model supports it  
        if (self.class_weight is not None and hasattr(self.model, 'set_params') and 'class_weight' in self.model.get_params()):
            self.model.set_params(class_weight = self.class_weight) # type: ignore
        
        return self.model.predict(X) # type: ignore


    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities if supported"""
        if hasattr(self.model, 'predict_proba'): # type: ignore

            return self.model.predict_proba(X) # type: ignore

        else:
            raise NotImplementedError("Model doesn't support probability predictions")
        
    def needs_label_encoding(self) -> bool:
        """Check if the model requires label encoding for targets"""
        return False
