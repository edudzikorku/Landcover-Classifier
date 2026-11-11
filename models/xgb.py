from typing import Dict, List, Any 
from xgboost import XGBClassifier # type: ignore
from .base_model import BaseModel 

class XGBClassificationModel(BaseModel):
    """XGBoost classification model implementation. Inherits from BaseModel."""
    def __init__(self, random_state: int = 42, n_splits: int = 10) -> None:
        super().__init__(random_state, n_splits = n_splits )
        self.model = XGBClassifier(random_state = random_state) # type: ignore

    def get_param_grid(self) -> Dict[str, List[Any]]:
        """Get the hyperparameter grid for the model"""

        return {
            'n_estimators': [10, 20],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'min_child_weight': [1, 5, 10],
            'gamma': [0, 0.2, 0.4],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'alpha': [0, 0.1, 1],
            'lambda': [0, 0.1, 1],
        }
    
    def needs_label_encoding(self):
        return True